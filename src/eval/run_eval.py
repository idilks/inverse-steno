"""Main evaluation loop.

Supports two modes:
  1. evaluate_from_files — original file-based evaluation (Condition 1)
  2. evaluate_benchmark — benchmark evaluation with attacker results (Condition 2)
"""

import json
from pathlib import Path

from ..schemes import SCHEMES
from ..detector import detect
from .auc import compute_auc


def evaluate_from_files(
    honest_path: str | Path,
    collusion_path: str | Path,
    candidates_path: str | Path,
) -> dict:
    """Run full evaluation from pre-generated data files.

    Expected JSON formats:
        honest/collusion: [{"task_id": str, "response": str, "question": str}, ...]
        candidates: {"task_id": [str, ...], ...}

    Returns:
        dict with AUC per scheme and overall
    """
    with open(honest_path) as f:
        honest_data = json.load(f)
    with open(collusion_path) as f:
        collusion_data = json.load(f)
    with open(candidates_path) as f:
        candidates_data = json.load(f)

    return _evaluate_items(honest_data, collusion_data, candidates_data)


def evaluate_benchmark(benchmark_results: dict) -> dict:
    """Evaluate benchmark results from run_benchmark.

    Handles both conditions:
      - Condition 1 (post_hoc): score distribution analysis
      - Condition 2 (redwood_attackers): AUC per attacker

    Args:
        benchmark_results: output from run_full_benchmark()

    Returns:
        dict with evaluation metrics per condition and attacker
    """
    output = {}

    # Condition 2: per-attacker AUC (already computed in run_benchmark,
    # but we recompute here for consistency with our eval pipeline)
    c2 = benchmark_results.get("condition2_redwood", {})
    per_attacker = {}
    for atk_name, atk_data in c2.get("per_attacker", {}).items():
        scores = atk_data.get("scores", [])
        labels = atk_data.get("labels", [])
        if scores and labels:
            auc = compute_auc(scores, labels)
            per_attacker[atk_name] = {
                "auc": auc,
                "avg_ber": atk_data.get("avg_ber", None),
                "n_samples": len(scores),
            }
    output["condition2"] = per_attacker

    # Condition 1: score statistics
    c1 = benchmark_results.get("condition1_post_hoc", {})
    scores = c1.get("scores", [])
    if scores:
        output["condition1"] = {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "n_samples": len(scores),
            "scheme_stats": c1.get("scheme_stats", {}),
        }

    return output


def _evaluate_items(
    honest_data: list[dict],
    collusion_data: list[dict],
    candidates_data: dict,
) -> dict:
    """Core evaluation logic shared by file-based and benchmark modes."""
    scores = []
    labels = []

    for item in honest_data:
        candidates = candidates_data.get(item["task_id"], [])
        result = detect(item["response"], candidates, SCHEMES)
        scores.append(result["score"])
        labels.append(0)

    for item in collusion_data:
        candidates = candidates_data.get(item["task_id"], [])
        result = detect(item["response"], candidates, SCHEMES)
        scores.append(result["score"])
        labels.append(1)

    overall_auc = compute_auc(scores, labels)

    # Per-scheme AUC
    per_scheme = {}
    for scheme_name in SCHEMES:
        scheme_scores = []
        scheme_labels = []
        for item in honest_data:
            candidates = candidates_data.get(item["task_id"], [])
            result = detect(item["response"], candidates, {scheme_name: SCHEMES[scheme_name]})
            scheme_scores.append(result["score"])
            scheme_labels.append(0)
        for item in collusion_data:
            candidates = candidates_data.get(item["task_id"], [])
            result = detect(item["response"], candidates, {scheme_name: SCHEMES[scheme_name]})
            scheme_scores.append(result["score"])
            scheme_labels.append(1)
        per_scheme[scheme_name] = compute_auc(scheme_scores, scheme_labels)

    return {
        "overall_auc": overall_auc,
        "per_scheme_auc": per_scheme,
        "n_honest": len(honest_data),
        "n_collusion": len(collusion_data),
    }
