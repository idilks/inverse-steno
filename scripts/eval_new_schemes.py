"""Test embedding + partition detectors on new schemes (keyword_presence, number_format)."""

import asyncio
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.detector.embedding import tfidf_embed, _knn_score
from src.detector.inverse_planning import detect
from src.schemes import SCHEMES
from src.schemes.partitions import PARTITIONS, filter_partitions
from src.schemes.keyword_presence import keyword_presence_encode
from src.schemes.number_format import number_format_encode
from src.eval.auc import compute_auc
from scripts.eval_diagnostic import load_prepared, rare_bit, simulate_attacker_response


def fast_embedding_pvalue(suspect_text, ref_texts, n_components=30, k=5):
    all_texts = ref_texts + [suspect_text]
    embs = tfidf_embed(all_texts, n=3, max_features=500)
    mean = embs.mean(axis=0)
    Xc = embs - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(n_components, Vt.shape[0], embs.shape[1])
    reduced = Xc @ Vt[:n_comp].T
    refs_r = reduced[:-1]
    sus_r = reduced[-1]
    sus_score = _knn_score(sus_r, refs_r, k=k)
    n = len(refs_r)
    null_scores = []
    for i in range(n):
        null_scores.append(_knn_score(refs_r[i], np.delete(refs_r, i, axis=0), k=k))
    rank = sum(1 for ns in null_scores if ns >= sus_score)
    return (rank + 1) / (n + 1)


async def main():
    prepared = await load_prepared(100, 100, "vertex_ai.gemini-3.1-flash-lite-preview", "test")
    print(f"Loaded {len(prepared)} questions", flush=True)

    test_schemes = {
        "keyword_presence": keyword_presence_encode,
        "number_format": number_format_encode,
    }

    rng = random.Random(42)
    ns = [1, 3, 5, 10]
    R = 5
    n_agents = 100

    for scheme_name, encode_fn in test_schemes.items():
        print(f"\n{'='*60}", flush=True)
        print(f"SCHEME: {scheme_name}", flush=True)
        print(f"{'='*60}", flush=True)

        pmins = []
        for q in prepared:
            bits = [encode_fn(c) for c in q["detector_refs"]]
            valid = [b for b in bits if b is not None]
            if valid:
                frac = sum(valid) / len(valid)
                pmins.append(min(frac, 1 - frac))
        print(f"p_min: mean={statistics.mean(pmins):.3f} "
              f"median={sorted(pmins)[len(pmins)//2]:.3f}", flush=True)

        per_q_emb_honest, per_q_emb_stego = [], []
        per_q_lo_honest, per_q_lo_stego = [], []

        for qi, q in enumerate(prepared):
            det_schemes = filter_partitions(q["detector_refs"], {**SCHEMES, **PARTITIONS})
            target = rare_bit(q["detector_refs"], encode_fn)

            emb_h, emb_s, lo_h, lo_s = [], [], [], []

            for _ in range(R):
                resp = rng.choice(q["attacker_pool"])
                p = fast_embedding_pvalue(resp, q["detector_refs"])
                emb_h.append(p)
                lo_r = detect(resp, q["detector_refs"], det_schemes)
                lo_h.append(lo_r["score"])

                stego = simulate_attacker_response(
                    q["attacker_pool"], encode_fn, target, 50, rng)
                if stego:
                    p = fast_embedding_pvalue(stego, q["detector_refs"])
                    emb_s.append(p)
                    lo_r = detect(stego, q["detector_refs"], det_schemes)
                    lo_s.append(lo_r["score"])
                else:
                    emb_s.append(None)
                    lo_s.append(None)

            per_q_emb_honest.append(emb_h)
            per_q_emb_stego.append(emb_s)
            per_q_lo_honest.append(lo_h)
            per_q_lo_stego.append(lo_s)

            if (qi + 1) % 10 == 0:
                print(f"  scored {qi+1}/{len(prepared)} questions", flush=True)

        valid_qs = [i for i in range(len(prepared))
                    if any(s is not None for s in per_q_emb_stego[i])]
        print(f"  valid questions: {len(valid_qs)}", flush=True)

        if not valid_qs:
            print("  SKIP — no valid stego responses (attacker can't find rare-bit candidates)")
            continue

        # Also run with the scheme as an oracle partition (upper bound)
        # to see what a partition detector COULD do if it knew the scheme
        per_q_oracle_honest, per_q_oracle_stego = [], []
        for qi, q in enumerate(prepared):
            oracle_schemes = {scheme_name: encode_fn}
            oh, os_ = [], []
            for _ in range(R):
                resp = rng.choice(q["attacker_pool"])
                r = detect(resp, q["detector_refs"], oracle_schemes)
                oh.append(r["score"])

                target = rare_bit(q["detector_refs"], encode_fn)
                stego = simulate_attacker_response(
                    q["attacker_pool"], encode_fn, target, 50, rng)
                if stego:
                    r = detect(stego, q["detector_refs"], oracle_schemes)
                    os_.append(r["score"])
                else:
                    os_.append(None)
            per_q_oracle_honest.append(oh)
            per_q_oracle_stego.append(os_)

        for n in ns:
            if len(valid_qs) < n:
                continue

            emb_scores, lo_scores, oracle_scores, labels = [], [], [], []
            rng_boot = random.Random(1000)

            for _ in range(n_agents):
                q_idxs = rng_boot.sample(range(len(prepared)), min(n, len(prepared)))
                emb_s = sum(-2.0 * math.log(max(rng_boot.choice(per_q_emb_honest[qi]), 1e-10))
                            for qi in q_idxs)
                lo_s = sum(rng_boot.choice(per_q_lo_honest[qi]) for qi in q_idxs)
                or_s = sum(rng_boot.choice(per_q_oracle_honest[qi]) for qi in q_idxs)
                emb_scores.append(emb_s)
                lo_scores.append(lo_s)
                oracle_scores.append(or_s)
                labels.append(0)

                q_idxs = rng_boot.sample(valid_qs, min(n, len(valid_qs)))
                emb_s, lo_s, or_s = 0.0, 0.0, 0.0
                for qi in q_idxs:
                    ve = [s for s in per_q_emb_stego[qi] if s is not None]
                    vl = [s for s in per_q_lo_stego[qi] if s is not None]
                    vo = [s for s in per_q_oracle_stego[qi] if s is not None]
                    if ve:
                        emb_s += -2.0 * math.log(max(rng_boot.choice(ve), 1e-10))
                    if vl:
                        lo_s += rng_boot.choice(vl)
                    if vo:
                        or_s += rng_boot.choice(vo)
                emb_scores.append(emb_s)
                lo_scores.append(lo_s)
                oracle_scores.append(or_s)
                labels.append(1)

            auc_emb = compute_auc(emb_scores, labels)
            auc_lo = compute_auc(lo_scores, labels)
            auc_or = compute_auc(oracle_scores, labels)
            print(f"  N={n:>2}: embedding={auc_emb:.3f}  "
                  f"log-odds(blind)={auc_lo:.3f}  "
                  f"oracle={auc_or:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
