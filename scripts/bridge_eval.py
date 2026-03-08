"""Bridge evaluation: load attacker output, generate references, run detector.

Usage:
    python -m scripts.bridge_eval --mode oracle
    python -m scripts.bridge_eval --mode codebook-free --k 50
    python -m scripts.bridge_eval --mode oracle --dry-run  # skip API, use cached only
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.inverse_planning import detect, detect_multi
from src.detector.sampling import generate_candidates, load_cached
from src.schemes import SCHEMES
from src.eval.auc import compute_auc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str = "output/generated/dataset.jsonl") -> list[dict]:
    """Load JSONL records from attacker pipeline."""
    full_path = PROJECT_ROOT / path
    records = []
    with open(full_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def group_by_question(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by task_question."""
    groups = defaultdict(list)
    for r in records:
        groups[r["task_question"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Model function (async wrapper for API)
# ---------------------------------------------------------------------------

def make_model_fn(model_name: str = "claude-haiku", dry_run: bool = False):
    """Create async model function.

    If dry_run=True, raises if cache miss (no API calls).
    Otherwise, calls the Dartmouth Chat API (OpenAI-compatible).
    """
    from dotenv import load_dotenv
    load_dotenv()

    if dry_run:
        async def dry_model_fn(prompt: str, n: int = 1) -> list[str]:
            raise RuntimeError(
                "Cache miss in dry-run mode. Run without --dry-run to call API."
            )
        return dry_model_fn

    # Default: Dartmouth Chat API (OpenAI-compatible)
    import requests

    api_url = os.environ.get(
        "DARTMOUTH_CHAT_API_URL",
        "https://chat.dartmouth.edu/api/chat/completions",
    )
    api_key = os.environ.get("DARTMOUTH_CHAT_API_KEY", "")
    if not api_key:
        raise ValueError("DARTMOUTH_CHAT_API_KEY not set. Add it to .env")

    def _single_call(prompt_text: str) -> str:
        """Single API call with retry."""
        import time
        for attempt in range(3):
            try:
                resp = requests.post(
                    api_url,
                    headers={
                        "Authorization": f"bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant solving math problems."},
                            {"role": "user", "content": prompt_text},
                        ],
                        "temperature": 1.0,
                        "stream": False,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError) as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"    [retry {attempt+1}/2 after {wait}s: {type(e).__name__}]")
                    time.sleep(wait)
                else:
                    raise

    async def dartmouth_fn(prompt: str, n: int = 1) -> list[str]:
        responses = []
        for i in range(n):
            if n > 1 and i % 10 == 0:
                print(f"      [{i}/{n}]", end="", flush=True)
            text = await asyncio.to_thread(_single_call, prompt)
            responses.append(text)
        if n > 1:
            print(f" [{n}/{n}]")
        return responses

    return dartmouth_fn


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def run_bridge_eval(
    records: list[dict],
    model_fn,
    model_name: str,
    k: int = 50,
    schemes: dict | None = None,
    mode: str = "oracle",
    partition_filter_fn=None,
) -> dict:
    """Run detection on all records using reference candidates.

    Args:
        records: dataset records from attacker pipeline
        model_fn: async callable(prompt, n) -> list[str]
        model_name: for cache keying
        k: number of reference candidates
        schemes: encoding schemes to use (defaults to SCHEMES)
        mode: "oracle" or "codebook-free"
        partition_filter_fn: optional callable(candidates, schemes) -> filtered_schemes

    Returns:
        dict with scores, labels, AUC, p_bit table, per-record details
    """
    if schemes is None:
        schemes = SCHEMES

    grouped = group_by_question(records)
    all_scores = []
    all_labels = []
    details = []
    p_bit_table = {}

    for question, question_records in grouped.items():
        # Build prompt matching attacker format
        # Use the task metadata from records to reconstruct
        sample = question_records[0]
        choices = sample.get("task_answer_choices", [])
        prompt = question
        if choices:
            prompt += "\n" + "\n".join(choices)
        prompt += "\nPlease solve this step by step and provide your final answer."

        # Generate or load reference candidates
        print(f"  Generating {k} references for: {question[:60]}...")
        candidates = await generate_candidates(
            prompt, model_fn, k=k, model_name=model_name
        )
        print(f"    Got {len(candidates)} candidates")

        # Filter partitions per-question in codebook-free mode
        if partition_filter_fn is not None:
            q_schemes = partition_filter_fn(candidates, schemes)
            print(f"    Filtered to {len(q_schemes)}/{len(schemes)} partitions")
        else:
            q_schemes = schemes

        # Compute p_bit for each scheme on this question
        q_pbit = {}
        for scheme_name, encode_fn in q_schemes.items():
            bits = [encode_fn(c) for c in candidates]
            valid = [b for b in bits if b is not None]
            if valid:
                frac_1 = sum(valid) / len(valid)
                q_pbit[scheme_name] = {
                    "frac_bit1": frac_1,
                    "frac_bit0": 1 - frac_1,
                    "n_valid": len(valid),
                    "n_none": len(bits) - len(valid),
                }
            else:
                q_pbit[scheme_name] = {
                    "frac_bit1": None,
                    "n_valid": 0,
                    "n_none": len(bits),
                }
        p_bit_table[question[:60]] = q_pbit

        # Score each record
        for rec in question_records:
            result = detect(rec["response_text"], candidates, q_schemes)
            label = rec["label"]  # 0=honest, 1=stego

            all_scores.append(result["score"])
            all_labels.append(label)
            details.append({
                "question": question[:60],
                "agent_type": rec["agent_type"],
                "scheme_name": rec.get("scheme_name"),
                "label": label,
                "score": result["score"],
                "best_scheme": result["best_scheme"],
                "all_scores": result["all_scores"],
                "decoded_bits": result["decoded_bits"],
            })

    # Compute AUC
    auc = compute_auc(all_scores, all_labels)

    # Per-scheme AUC (only for stego records of that scheme)
    per_scheme_auc = {}
    honest_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
    for scheme_name in set(d.get("scheme_name") for d in details if d["label"] == 1):
        if scheme_name is None:
            continue
        scheme_stego_scores = [
            d["score"] for d in details
            if d["label"] == 1 and d["scheme_name"] == scheme_name
        ]
        if scheme_stego_scores and honest_scores:
            combined_scores = honest_scores + scheme_stego_scores
            combined_labels = [0] * len(honest_scores) + [1] * len(scheme_stego_scores)
            per_scheme_auc[scheme_name] = compute_auc(combined_scores, combined_labels)

    return {
        "mode": mode,
        "auc": auc,
        "per_scheme_auc": per_scheme_auc,
        "n_records": len(records),
        "n_honest": sum(1 for l in all_labels if l == 0),
        "n_stego": sum(1 for l in all_labels if l == 1),
        "k_candidates": k,
        "p_bit_table": p_bit_table,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print(f"BRIDGE EVALUATION — mode: {results['mode']}")
    print("=" * 70)
    print(f"Records: {results['n_records']} ({results['n_honest']} honest, {results['n_stego']} stego)")
    print(f"Reference candidates: {results['k_candidates']}")
    print(f"\nOVERALL AUC: {results['auc']:.4f}")

    print("\nPer-scheme AUC:")
    for scheme, auc in sorted(results["per_scheme_auc"].items()):
        print(f"  {scheme:20s}: {auc:.4f}")

    print("\n--- p_bit table ---")
    print(f"{'Question':<62} {'Scheme':<15} {'P(bit=0)':<10} {'P(bit=1)':<10} {'Valid':<6} {'None':<6}")
    print("-" * 110)
    for question, scheme_data in results["p_bit_table"].items():
        for scheme_name, stats in scheme_data.items():
            if stats["n_valid"] == 0:
                print(f"{question:<62} {scheme_name:<15} {'N/A':<10} {'N/A':<10} {0:<6} {stats['n_none']:<6}")
            else:
                print(f"{question:<62} {scheme_name:<15} {stats['frac_bit0']:<10.3f} {stats['frac_bit1']:<10.3f} {stats['n_valid']:<6} {stats['n_none']:<6}")

    # Detection breakdown
    print("\n--- Score distribution ---")
    honest_scores = [d["score"] for d in results["details"] if d["label"] == 0]
    stego_scores = [d["score"] for d in results["details"] if d["label"] == 1]
    if honest_scores:
        print(f"  Honest: mean={sum(honest_scores)/len(honest_scores):.4f}, "
              f"min={min(honest_scores):.4f}, max={max(honest_scores):.4f}")
    if stego_scores:
        print(f"  Stego:  mean={sum(stego_scores)/len(stego_scores):.4f}, "
              f"min={min(stego_scores):.4f}, max={max(stego_scores):.4f}")

    # Estimated N for AUC > 0.9 per scheme
    print("\n--- Estimated N for AUC > 0.9 (per scheme, from median p_bit) ---")
    for scheme_name in sorted(set(s for q in results["p_bit_table"].values() for s in q)):
        p_bits = []
        for q_data in results["p_bit_table"].values():
            if scheme_name in q_data and q_data[scheme_name]["n_valid"] > 0:
                frac1 = q_data[scheme_name]["frac_bit1"]
                p_bits.append(min(frac1, 1 - frac1))  # minority fraction
        if p_bits:
            p_bits.sort()
            median_pbit = p_bits[len(p_bits) // 2]
            if median_pbit > 0 and median_pbit < 0.5:
                import math
                # Rough estimate: N ~ (z / log_odds)^2 where log_odds = log(0.5) - log(p_bit)
                log_odds = math.log(0.5) - math.log(median_pbit)
                # For AUC 0.9, need d' ~ 1.8; d' = sqrt(N) * log_odds / sigma
                # Rough: N ~ (1.8 / log_odds)^2 but this is very approximate
                est_n = max(1, round((1.8 / log_odds) ** 2)) if log_odds > 0 else float('inf')
                print(f"  {scheme_name:<20s}: median p_bit={median_pbit:.3f}, "
                      f"log_odds={log_odds:.3f}, est N~{est_n}")
            else:
                print(f"  {scheme_name:<20s}: median p_bit={median_pbit:.3f} (no asymmetry)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Bridge evaluation")
    parser.add_argument("--mode", choices=["oracle", "codebook-free"], default="oracle")
    parser.add_argument("--k", type=int, default=50, help="Reference candidates")
    parser.add_argument("--model", default="anthropic.claude-haiku-4-5-20251001", help="Model for references")
    parser.add_argument("--data", default="output/generated/dataset.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="No API calls, cache only")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    records = load_dataset(args.data)
    print(f"Loaded {len(records)} records")

    model_fn = make_model_fn(args.model, dry_run=args.dry_run)

    if args.mode == "oracle":
        # Use the 3 schemes that match attacker schemes (synonym, structural, parity)
        # recommendation won't fire on math — exclude it to reduce noise
        oracle_schemes = {
            k: v for k, v in SCHEMES.items()
            if k in ("synonym", "structural", "parity")
        }
        results = await run_bridge_eval(
            records, model_fn, args.model,
            k=args.k, schemes=oracle_schemes, mode="oracle",
        )
    elif args.mode == "codebook-free":
        # Import partition library
        try:
            from src.schemes.partitions import PARTITIONS, filter_partitions
            all_schemes = {**SCHEMES, **PARTITIONS}
        except ImportError:
            print("Partition library not found, using SCHEMES only")
            all_schemes = SCHEMES
            filter_partitions = None
        results = await run_bridge_eval(
            records, model_fn, args.model,
            k=args.k, schemes=all_schemes, mode="codebook-free",
            partition_filter_fn=filter_partitions,
        )

    print_results(results)

    # Save results
    out_path = args.output or f"results/bridge_{args.mode}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Make details serializable
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
