"""Permutation testing for multi-partition detection.

When testing M partitions simultaneously, we need to correct for multiple
comparisons. The key statistic is max_j S_j — the maximum detection score
across all partitions. Under H0 (no steganography), this has a distribution
we estimate by permutation.
"""

import random
import math
from typing import Callable

from ..detector.inverse_planning import detect
from .auc import compute_auc


def permutation_null(
    responses: list[str],
    candidates: list[str],
    schemes: dict[str, Callable[[str], int | None]],
    n_perms: int = 10000,
    seed: int = 42,
) -> list[float]:
    """Estimate null distribution of max_j S_j by permuting response labels.

    Under H0, all responses come from the same (honest) distribution.
    We permute which responses are "tested" vs "reference" and compute
    the max detection score across schemes.

    Args:
        responses: all responses (honest + stego mixed)
        candidates: reference candidates
        schemes: partition/scheme dict
        n_perms: number of permutations
        seed: random seed

    Returns:
        sorted list of n_perms max-scores under H0
    """
    rng = random.Random(seed)
    null_maxes = []

    for _ in range(n_perms):
        # Pick a random response to "test"
        idx = rng.randrange(len(responses))
        test_resp = responses[idx]

        # Use remaining as pseudo-candidates (or keep original candidates)
        result = detect(test_resp, candidates, schemes)
        null_maxes.append(result["score"])

    null_maxes.sort()
    return null_maxes


def permutation_threshold(
    scores_honest: list[float],
    scores_stego: list[float],
    n_perms: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute detection threshold via label permutation.

    Permutes honest/stego labels to build null distribution of the
    AUC statistic (or score separation), then finds the threshold
    that controls false positive rate at alpha.

    Args:
        scores_honest: detection scores for honest responses
        scores_stego: detection scores for stego responses
        n_perms: number of permutations
        alpha: significance level
        seed: random seed

    Returns:
        dict with threshold, null_distribution, p_value of observed separation
    """
    rng = random.Random(seed)
    all_scores = scores_honest + scores_stego
    n_honest = len(scores_honest)
    n_total = len(all_scores)

    # Observed test statistic: difference in means
    observed_diff = (
        (sum(scores_stego) / len(scores_stego) if scores_stego else 0)
        - (sum(scores_honest) / len(scores_honest) if scores_honest else 0)
    )

    null_diffs = []
    for _ in range(n_perms):
        perm = list(all_scores)
        rng.shuffle(perm)
        perm_honest = perm[:n_honest]
        perm_stego = perm[n_honest:]
        diff = (
            (sum(perm_stego) / len(perm_stego) if perm_stego else 0)
            - (sum(perm_honest) / len(perm_honest) if perm_honest else 0)
        )
        null_diffs.append(diff)

    null_diffs.sort()

    # p-value: fraction of null diffs >= observed
    n_extreme = sum(1 for d in null_diffs if d >= observed_diff)
    p_value = n_extreme / n_perms

    # Threshold at alpha
    threshold_idx = int((1 - alpha) * n_perms)
    threshold = null_diffs[min(threshold_idx, len(null_diffs) - 1)]

    return {
        "observed_diff": observed_diff,
        "threshold": threshold,
        "p_value": p_value,
        "alpha": alpha,
        "n_perms": n_perms,
        "null_distribution": null_diffs,
    }


def benjamini_hochberg(p_values: list[tuple[str, float]], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg FDR correction for M partitions.

    Args:
        p_values: list of (partition_name, raw_p_value)
        alpha: target FDR

    Returns:
        list of dicts with partition, raw_p, adjusted_p, significant
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value
    sorted_pvals = sorted(p_values, key=lambda x: x[1])

    results = []
    for rank, (name, p) in enumerate(sorted_pvals, 1):
        # BH adjusted p-value
        adjusted = p * m / rank
        results.append({
            "partition": name,
            "raw_p": p,
            "rank": rank,
            "bh_threshold": alpha * rank / m,
            "adjusted_p": min(adjusted, 1.0),
        })

    # Enforce monotonicity (adjusted p can't decrease with rank)
    for i in range(len(results) - 2, -1, -1):
        results[i]["adjusted_p"] = min(results[i]["adjusted_p"], results[i + 1]["adjusted_p"])

    # Mark significant
    for r in results:
        r["significant"] = r["adjusted_p"] <= alpha

    return results


def partition_pvalues(
    response: str,
    candidates: list[str],
    schemes: dict[str, Callable[[str], int | None]],
    n_perms: int = 1000,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Compute a permutation p-value for each partition.

    For each partition, estimate how extreme the observed score is
    under the null (random assignment of response among candidates).

    Args:
        response: the response to test
        candidates: reference candidates
        schemes: partition dict
        n_perms: permutations per partition
        seed: random seed

    Returns:
        list of (partition_name, p_value)
    """
    rng = random.Random(seed)
    result = detect(response, candidates, schemes)
    observed_scores = result["all_scores"]

    p_values = []
    for scheme_name, encode_fn in schemes.items():
        observed = observed_scores.get(scheme_name, 0.0)

        # Null: score random candidates against the rest
        null_scores = []
        for _ in range(n_perms):
            idx = rng.randrange(len(candidates))
            pseudo_resp = candidates[idx]
            pseudo_others = candidates[:idx] + candidates[idx + 1:]
            pseudo_result = detect(pseudo_resp, pseudo_others, {scheme_name: encode_fn})
            null_scores.append(pseudo_result["score"])

        # p-value: fraction of null >= observed
        n_extreme = sum(1 for s in null_scores if s >= observed)
        p_val = (n_extreme + 1) / (n_perms + 1)  # continuity correction
        p_values.append((scheme_name, p_val))

    return p_values
