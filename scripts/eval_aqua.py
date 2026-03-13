"""Unified AQUA-RAT evaluation: single-obs AUC, AUC-vs-k, multi-obs AUC-vs-N.

Loads AQUA-RAT questions, consumes pre-cached candidates (from generate_aqua_candidates.py),
simulates attacker responses from the candidate pool, runs all three experiments.

Usage:
    python -m scripts.eval_aqua --experiment all --n-questions 100 --k-ref 100
    python -m scripts.eval_aqua --experiment single --codebook-free
    python -m scripts.eval_aqua --experiment auc-vs-k --attacker-ks 1 2 5 10 20 50
    python -m scripts.eval_aqua --experiment multi-obs --max-n 10
    python -m scripts.eval_aqua --n-questions 3 --dry-run  # test with cached only
"""

import argparse
import asyncio
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.dataset import load_aqua_rat, format_question
from src.detector.inverse_planning import detect
from src.detector.sampling import generate_candidates
from src.schemes import SCHEMES
from src.schemes.partitions import PARTITIONS, filter_partitions
from src.eval.auc import compute_auc
from scripts.bridge_eval import make_model_fn

PROMPT_SUFFIX = "\nPlease solve this step by step and provide your final answer."


# ---------------------------------------------------------------------------
# Attacker simulation (inlined from experiment_auc_vs_k for self-containment)
# ---------------------------------------------------------------------------

def simulate_attacker_response(
    candidates: list[str],
    encode_fn,
    target_bit: int,
    attacker_k: int,
    rng: random.Random,
) -> str | None:
    """Simulate attacker with rejection sampling budget k."""
    pool = list(candidates) if attacker_k >= len(candidates) else rng.sample(candidates, attacker_k)
    compatible = [c for c in pool if encode_fn(c) == target_bit]
    return compatible[0] if compatible else None


def rare_bit(candidates: list[str], encode_fn) -> int:
    """Return the minority bit among candidates."""
    bits = [encode_fn(c) for c in candidates if encode_fn(c) is not None]
    if not bits:
        return 0
    return 1 if sum(bits) / len(bits) <= 0.5 else 0


# ---------------------------------------------------------------------------
# Data loading & candidate prep
# ---------------------------------------------------------------------------

async def load_questions_and_candidates(
    n_questions: int,
    k_ref: int,
    model_fn,
    model_name: str,
    split: str = "test",
) -> list[dict]:
    """Load AQUA-RAT questions and their cached candidates.

    Returns list of dicts with keys: task_id, question, prompt, detector_refs, attacker_pool
    """
    questions = load_aqua_rat(n=n_questions, split=split)
    print(f"Loaded {len(questions)} AQUA-RAT questions")

    prepared = []
    for item in questions:
        prompt = format_question(item) + PROMPT_SUFFIX

        candidates = await generate_candidates(
            prompt, model_fn, k=k_ref, model_name=model_name
        )

        if len(candidates) < 4:
            print(f"  SKIP {item['task_id']}: only {len(candidates)} candidates")
            continue

        split_idx = len(candidates) // 2
        prepared.append({
            "task_id": item["task_id"],
            "question": item["question"],
            "prompt": prompt,
            "detector_refs": candidates[:split_idx],
            "attacker_pool": candidates[split_idx:],
        })

    print(f"Prepared {len(prepared)}/{len(questions)} questions "
          f"({k_ref} candidates each, split 50/50)")
    return prepared


# ---------------------------------------------------------------------------
# p_bit diagnostics
# ---------------------------------------------------------------------------

def compute_p_bit_table(
    prepared: list[dict],
    schemes: dict,
) -> dict:
    """Compute p_bit for each question × scheme on detector refs."""
    table = {}
    for q in prepared:
        q_pbit = {}
        for scheme_name, encode_fn in schemes.items():
            bits = [encode_fn(c) for c in q["detector_refs"]]
            valid = [b for b in bits if b is not None]
            if valid:
                frac_1 = sum(valid) / len(valid)
                q_pbit[scheme_name] = {
                    "frac_bit0": 1 - frac_1,
                    "frac_bit1": frac_1,
                    "n_valid": len(valid),
                }
            else:
                q_pbit[scheme_name] = {"frac_bit0": None, "frac_bit1": None, "n_valid": 0}
        table[q["task_id"]] = q_pbit
    return table


def print_p_bit_summary(p_bit_table: dict, schemes: dict):
    """Print median p_bit per scheme across questions."""
    print("\n--- p_bit summary (median minority fraction per scheme) ---")
    for scheme_name in sorted(schemes):
        minorities = []
        for q_data in p_bit_table.values():
            if scheme_name in q_data and q_data[scheme_name]["n_valid"] > 0:
                f1 = q_data[scheme_name]["frac_bit1"]
                minorities.append(min(f1, 1 - f1))
        if minorities:
            minorities.sort()
            med = minorities[len(minorities) // 2]
            print(f"  {scheme_name:<25s}: median p_min={med:.3f}  (n={len(minorities)} questions)")
        else:
            print(f"  {scheme_name:<25s}: no valid data")


# ---------------------------------------------------------------------------
# Experiment 1: Single-observation AUC
# ---------------------------------------------------------------------------

def run_single_obs(
    prepared: list[dict],
    schemes: dict,
    rng: random.Random,
    codebook_free: bool = False,
    attacker_k: int = 50,
) -> dict:
    """Single-observation AUC, per-scheme and overall."""
    oracle_schemes = {k: v for k, v in SCHEMES.items() if k in ("synonym", "structural", "parity")}

    per_scheme = {}
    overall_scores = []
    overall_labels = []

    for scheme_name, encode_fn in oracle_schemes.items():
        scores = []
        labels = []

        for q in prepared:
            # Select detection schemes for this question
            if codebook_free:
                all_s = {**SCHEMES, **PARTITIONS}
                det_schemes = filter_partitions(q["detector_refs"], all_s)
            else:
                det_schemes = oracle_schemes

            # Honest: random sample from attacker pool, no bit constraint
            honest_resp = rng.choice(q["attacker_pool"])
            result = detect(honest_resp, q["detector_refs"], det_schemes)
            scores.append(result["score"])
            labels.append(0)
            overall_scores.append(result["score"])
            overall_labels.append(0)

            # Stego: target rare bit
            target = rare_bit(q["detector_refs"], encode_fn)
            stego_resp = simulate_attacker_response(
                q["attacker_pool"], encode_fn, target, attacker_k, rng
            )
            if stego_resp is None:
                continue
            result = detect(stego_resp, q["detector_refs"], det_schemes)
            scores.append(result["score"])
            labels.append(1)
            overall_scores.append(result["score"])
            overall_labels.append(1)

        auc = compute_auc(scores, labels) if sum(labels) > 0 else 0.5
        n_stego = sum(labels)
        per_scheme[scheme_name] = {"auc": auc, "n_honest": len(labels) - n_stego, "n_stego": n_stego}

    overall_auc = compute_auc(overall_scores, overall_labels) if overall_labels else 0.5

    return {
        "experiment": "single_obs",
        "codebook_free": codebook_free,
        "overall_auc": overall_auc,
        "per_scheme": per_scheme,
        "n_questions": len(prepared),
    }


# ---------------------------------------------------------------------------
# Experiment 2: AUC vs attacker k
# ---------------------------------------------------------------------------

def run_auc_vs_k(
    prepared: list[dict],
    schemes: dict,
    rng: random.Random,
    attacker_ks: list[int] = None,
    codebook_free: bool = False,
) -> dict:
    """AUC as a function of attacker rejection-sampling budget k."""
    if attacker_ks is None:
        attacker_ks = [1, 2, 5, 10, 20, 50]

    oracle_schemes = {k: v for k, v in SCHEMES.items() if k in ("synonym", "structural", "parity")}
    results_per_k = {}

    for ak in attacker_ks:
        print(f"\n--- attacker k = {ak} ---")
        per_scheme = {}

        for scheme_name, encode_fn in oracle_schemes.items():
            scores = []
            labels = []

            for q in prepared:
                if codebook_free:
                    all_s = {**SCHEMES, **PARTITIONS}
                    det_schemes = filter_partitions(q["detector_refs"], all_s)
                else:
                    det_schemes = oracle_schemes

                # Honest
                honest_resp = rng.choice(q["attacker_pool"])
                result = detect(honest_resp, q["detector_refs"], det_schemes)
                scores.append(result["score"])
                labels.append(0)

                # Stego
                target = rare_bit(q["detector_refs"], encode_fn)
                stego_resp = simulate_attacker_response(
                    q["attacker_pool"], encode_fn, target, ak, rng
                )
                if stego_resp is None:
                    continue
                result = detect(stego_resp, q["detector_refs"], det_schemes)
                scores.append(result["score"])
                labels.append(1)

            auc = compute_auc(scores, labels) if sum(labels) > 0 else 0.5
            n_stego = sum(labels)
            per_scheme[scheme_name] = {"auc": auc, "n_stego": n_stego, "n_failed": len(prepared) - n_stego}
            print(f"  {scheme_name}: AUC={auc:.4f} (stego={n_stego}, failed={len(prepared) - n_stego})")

        # Overall
        all_scores = []
        all_labels = []
        for scheme_name, encode_fn in oracle_schemes.items():
            for q in prepared:
                if codebook_free:
                    all_s = {**SCHEMES, **PARTITIONS}
                    det_schemes = filter_partitions(q["detector_refs"], all_s)
                else:
                    det_schemes = oracle_schemes

                # One honest per question (shared across schemes)
                if scheme_name == list(oracle_schemes.keys())[0]:
                    honest_resp = rng.choice(q["attacker_pool"])
                    result = detect(honest_resp, q["detector_refs"], det_schemes)
                    all_scores.append(result["score"])
                    all_labels.append(0)

                target = rare_bit(q["detector_refs"], encode_fn)
                stego_resp = simulate_attacker_response(
                    q["attacker_pool"], encode_fn, target, ak, rng
                )
                if stego_resp:
                    result = detect(stego_resp, q["detector_refs"], det_schemes)
                    all_scores.append(result["score"])
                    all_labels.append(1)

        overall_auc = compute_auc(all_scores, all_labels) if all_labels else 0.5
        results_per_k[ak] = {"overall_auc": overall_auc, "per_scheme": per_scheme}
        print(f"  OVERALL: {overall_auc:.4f}")

    return {
        "experiment": "auc_vs_k",
        "codebook_free": codebook_free,
        "attacker_ks": attacker_ks,
        "results": results_per_k,
        "n_questions": len(prepared),
    }


# ---------------------------------------------------------------------------
# Experiment 3: Multi-observation AUC vs N
# ---------------------------------------------------------------------------

def run_multi_obs(
    prepared: list[dict],
    schemes: dict,
    rng: random.Random,
    ns: list[int] = None,
    n_agents: int = 50,
    attacker_k: int = 50,
    codebook_free: bool = False,
) -> dict:
    """AUC as a function of number of observations N (questions per agent).

    Simulates n_agents honest agents and n_agents stego agents.
    Each agent answers N questions. Sum log-odds across N, then compute AUC.
    """
    if ns is None:
        ns = [1, 2, 3, 5, 7, 10]

    oracle_schemes = {k: v for k, v in SCHEMES.items() if k in ("synonym", "structural", "parity")}
    max_n = max(ns)

    if len(prepared) < max_n:
        print(f"  WARNING: only {len(prepared)} questions, capping max_n")
        ns = [n for n in ns if n <= len(prepared)]

    # Precompute per-question, per-PARTITION scores for honest and stego.
    # Key fix: store all_scores (dict of partition -> score) per question,
    # so we can sum per-partition across observations THEN take max.
    # This is max_j sum_i s(i,j) — not sum_i max_j s(i,j).

    # Collect the union of all detection partitions across questions
    # (in codebook-free mode, each question may have a different filtered set)
    per_q_det_schemes = []  # det_schemes used for each question
    per_q_honest_all_scores = []  # list of {partition: score} dicts
    per_q_stego_all_scores = {}   # attacker_scheme -> list of {partition: score} or None

    for scheme_name in oracle_schemes:
        per_q_stego_all_scores[scheme_name] = []

    for qi, q in enumerate(prepared):
        if codebook_free:
            all_s = {**SCHEMES, **PARTITIONS}
            det_schemes = filter_partitions(q["detector_refs"], all_s)
        else:
            det_schemes = oracle_schemes
        per_q_det_schemes.append(det_schemes)

        # Honest: store per-partition scores
        honest_resp = rng.choice(q["attacker_pool"])
        result = detect(honest_resp, q["detector_refs"], det_schemes)
        per_q_honest_all_scores.append(result["all_scores"])

        # Stego: store per-partition scores for each attacker scheme
        for scheme_name, encode_fn in oracle_schemes.items():
            target = rare_bit(q["detector_refs"], encode_fn)
            stego_resp = simulate_attacker_response(
                q["attacker_pool"], encode_fn, target, attacker_k, rng
            )
            if stego_resp is None:
                per_q_stego_all_scores[scheme_name].append(None)
            else:
                result = detect(stego_resp, q["detector_refs"], det_schemes)
                per_q_stego_all_scores[scheme_name].append(result["all_scores"])

    def _agent_score(question_indices, per_q_all_scores_list):
        """Sum per-partition across questions, then take max. = max_j sum_i s(i,j)."""
        partition_sums = {}
        for qi in question_indices:
            all_scores = per_q_all_scores_list[qi]
            if all_scores is None:
                continue
            for part, s in all_scores.items():
                partition_sums[part] = partition_sums.get(part, 0.0) + s
        return max(partition_sums.values()) if partition_sums else 0.0

    # Bootstrap: for each N, sample N questions per agent, sum-then-max
    results_by_n = {}
    for n in ns:
        per_scheme_auc = {}

        for scheme_name in oracle_schemes:
            scores = []
            labels = []
            valid_qs = [i for i, s in enumerate(per_q_stego_all_scores[scheme_name]) if s is not None]

            if len(valid_qs) < n:
                per_scheme_auc[scheme_name] = {"auc": 0.5, "note": "insufficient questions"}
                continue

            for _ in range(n_agents):
                # Honest agent: sample N questions, sum-then-max
                q_idxs = rng.sample(range(len(prepared)), min(n, len(prepared)))
                agent_score = _agent_score(q_idxs, per_q_honest_all_scores)
                scores.append(agent_score)
                labels.append(0)

                # Stego agent: sample N questions from valid set, sum-then-max
                q_idxs = rng.sample(valid_qs, min(n, len(valid_qs)))
                agent_score = _agent_score(q_idxs, per_q_stego_all_scores[scheme_name])
                scores.append(agent_score)
                labels.append(1)

            auc = compute_auc(scores, labels)
            per_scheme_auc[scheme_name] = {"auc": auc}

        results_by_n[n] = per_scheme_auc
        scheme_aucs = " | ".join(f"{s}={d['auc']:.3f}" for s, d in per_scheme_auc.items())
        print(f"  N={n:>2}: {scheme_aucs}")

    return {
        "experiment": "multi_obs",
        "codebook_free": codebook_free,
        "ns": ns,
        "n_agents": n_agents,
        "attacker_k": attacker_k,
        "results": results_by_n,
        "n_questions": len(prepared),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_single_obs(results: dict):
    print("\n" + "=" * 70)
    mode = "codebook-free" if results["codebook_free"] else "oracle"
    print(f"SINGLE-OBSERVATION AUC ({mode}, {results['n_questions']} questions)")
    print("=" * 70)
    print(f"{'Scheme':<20} {'AUC':<10} {'N_honest':<10} {'N_stego':<10}")
    print("-" * 50)
    for scheme, data in sorted(results["per_scheme"].items()):
        print(f"{scheme:<20} {data['auc']:<10.4f} {data['n_honest']:<10} {data['n_stego']:<10}")
    print(f"\n{'OVERALL':<20} {results['overall_auc']:<10.4f}")


def print_auc_vs_k(results: dict):
    print("\n" + "=" * 70)
    mode = "codebook-free" if results["codebook_free"] else "oracle"
    print(f"AUC vs ATTACKER-k ({mode}, {results['n_questions']} questions)")
    print("=" * 70)

    scheme_names = sorted(next(iter(results["results"].values()))["per_scheme"].keys())
    header = f"{'k':<6} {'Overall':<10}"
    for s in scheme_names:
        header += f"{s:<14}"
    print(header)
    print("-" * len(header))

    for ak in results["attacker_ks"]:
        r = results["results"][ak]
        line = f"{ak:<6} {r['overall_auc']:<10.4f}"
        for s in scheme_names:
            line += f"{r['per_scheme'][s]['auc']:<14.4f}"
        print(line)


def print_multi_obs(results: dict):
    print("\n" + "=" * 70)
    mode = "codebook-free" if results["codebook_free"] else "oracle"
    print(f"MULTI-OBSERVATION AUC vs N ({mode}, {results['n_questions']} questions, "
          f"{results['n_agents']} agents)")
    print("=" * 70)

    scheme_names = sorted(next(iter(results["results"].values())).keys())
    header = f"{'N':<6}"
    for s in scheme_names:
        header += f"{s:<14}"
    print(header)
    print("-" * len(header))

    for n in results["ns"]:
        line = f"{n:<6}"
        for s in scheme_names:
            auc = results["results"][n].get(s, {}).get("auc", float("nan"))
            line += f"{auc:<14.4f}"
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="AQUA-RAT evaluation pipeline")
    parser.add_argument("--experiment", choices=["single", "auc-vs-k", "multi-obs", "all"],
                        default="all")
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--k-ref", type=int, default=100, help="Reference candidates per question")
    parser.add_argument("--model", default="anthropic.claude-haiku-4-5-20251001")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dry-run", action="store_true", help="No API calls, cache only")
    parser.add_argument("--codebook-free", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output JSON path")

    # AUC-vs-k params
    parser.add_argument("--attacker-ks", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])

    # Multi-obs params
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=50, help="Simulated agents per condition")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    model_fn = make_model_fn(args.model, dry_run=args.dry_run)

    # Load data
    prepared = await load_questions_and_candidates(
        args.n_questions, args.k_ref, model_fn, args.model, args.split
    )

    if not prepared:
        print("No questions with cached candidates. Run generate_aqua_candidates.py first.")
        return

    # Schemes
    schemes = {k: v for k, v in SCHEMES.items() if k in ("synonym", "structural", "parity")}

    # p_bit diagnostics
    p_bit_table = compute_p_bit_table(prepared, schemes)
    print_p_bit_summary(p_bit_table, schemes)

    # Run experiments
    all_results = {"n_questions": len(prepared), "k_ref": args.k_ref, "model": args.model}

    if args.experiment in ("single", "all"):
        print("\n>>> Running single-observation AUC...")
        single = run_single_obs(prepared, schemes, rng, codebook_free=args.codebook_free)
        print_single_obs(single)
        all_results["single_obs"] = single

    if args.experiment in ("auc-vs-k", "all"):
        print("\n>>> Running AUC vs attacker-k...")
        auc_k = run_auc_vs_k(prepared, schemes, rng,
                              attacker_ks=args.attacker_ks,
                              codebook_free=args.codebook_free)
        print_auc_vs_k(auc_k)
        all_results["auc_vs_k"] = auc_k

    if args.experiment in ("multi-obs", "all"):
        print("\n>>> Running multi-observation AUC vs N...")
        ns = [n for n in [1, 2, 3, 5, 7, 10] if n <= args.max_n]
        multi = run_multi_obs(prepared, schemes, rng,
                               ns=ns, n_agents=args.n_agents,
                               codebook_free=args.codebook_free)
        print_multi_obs(multi)
        all_results["multi_obs"] = multi

    # Save
    out_path = args.output or f"results/aqua_{args.experiment}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
