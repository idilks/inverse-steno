"""Validate partitions on reference candidates.

Usage:
    python -m scripts.validate_partitions --data output/generated/dataset.jsonl

Runs every partition on the attacker dataset responses and prints split ratios.
When reference candidates are available (from bridge_eval cache), uses those too.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemes import SCHEMES
from src.schemes.partitions import PARTITIONS


def validate_on_responses(responses: list[str], partitions: dict) -> list[dict]:
    """Run all partitions on a set of responses. Return stats."""
    results = []
    for name, fn in sorted(partitions.items()):
        bits = [fn(r) for r in responses]
        n_none = sum(1 for b in bits if b is None)
        valid = [b for b in bits if b is not None]
        n_valid = len(valid)
        if n_valid == 0:
            results.append({
                "name": name,
                "n_valid": 0,
                "n_none": n_none,
                "frac_bit0": None,
                "frac_bit1": None,
                "status": "DEAD",
            })
            continue
        frac_1 = sum(valid) / n_valid
        frac_0 = 1 - frac_1

        # Check quality
        if n_none / len(bits) > 0.9:
            status = "HIGH_NONE"
        elif max(frac_0, frac_1) > 0.95:
            status = "ONE_SIDED"
        elif 0.1 <= min(frac_0, frac_1):
            status = "GOOD"
        else:
            status = "WEAK"

        results.append({
            "name": name,
            "n_valid": n_valid,
            "n_none": n_none,
            "frac_bit0": frac_0,
            "frac_bit1": frac_1,
            "status": status,
        })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="output/generated/dataset.jsonl")
    args = parser.parse_args()

    # Load responses
    data_path = PROJECT_ROOT / args.data
    responses = []
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            responses.append(rec["response_text"])

    print(f"Loaded {len(responses)} responses from {args.data}")
    print()

    # Run SCHEMES first
    print("=== SCHEMES (detector's known schemes) ===")
    scheme_results = validate_on_responses(responses, SCHEMES)
    _print_table(scheme_results)

    # Run PARTITIONS
    print(f"\n=== PARTITIONS ({len(PARTITIONS)} registered) ===")
    part_results = validate_on_responses(responses, PARTITIONS)
    _print_table(part_results)

    # Summary
    good = [r for r in part_results if r["status"] == "GOOD"]
    weak = [r for r in part_results if r["status"] == "WEAK"]
    dead = [r for r in part_results if r["status"] in ("DEAD", "HIGH_NONE", "ONE_SIDED")]
    print(f"\nSummary: {len(good)} GOOD, {len(weak)} WEAK, {len(dead)} dead/one-sided")
    print(f"Usable partitions (GOOD + WEAK): {len(good) + len(weak)}")


def _print_table(results):
    print(f"{'Name':<35} {'Valid':<6} {'None':<6} {'P(0)':<8} {'P(1)':<8} {'Status':<10}")
    print("-" * 80)
    for r in results:
        if r["frac_bit0"] is not None:
            print(f"{r['name']:<35} {r['n_valid']:<6} {r['n_none']:<6} "
                  f"{r['frac_bit0']:<8.3f} {r['frac_bit1']:<8.3f} {r['status']:<10}")
        else:
            print(f"{r['name']:<35} {r['n_valid']:<6} {r['n_none']:<6} "
                  f"{'N/A':<8} {'N/A':<8} {r['status']:<10}")


if __name__ == "__main__":
    main()
