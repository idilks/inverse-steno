"""Main evaluation loop."""

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
