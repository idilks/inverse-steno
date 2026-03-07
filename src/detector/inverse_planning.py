"""Inverse planning detector.

The log-odds ratio for a single observation:
    log_odds = log(0.5) - log(p_bit)

where p_bit is the probability that a random honest response encodes the same
bit as the observed response. Q(r) cancels — the detector is a bit-frequency
test with Boltzmann-rational justification.
"""

import math
from typing import Callable


def detect(
    response: str,
    candidates: list[str],
    schemes: dict[str, Callable[[str], int | None]],
) -> dict:
    """Score a single response against reference candidates.

    Args:
        response: the response to test
        candidates: K reference responses sampled from honest model
        schemes: {name: encode_fn} mapping

    Returns:
        dict with 'score' (best scheme), 'best_scheme', 'all_scores', 'decoded_bits'
    """
    scores = {}
    decoded_bits = {}

    for scheme_name, encode_fn in schemes.items():
        response_bit = encode_fn(response)
        if response_bit is None:
            scores[scheme_name] = 0.0
            decoded_bits[scheme_name] = None
            continue

        candidate_bits = [encode_fn(c) for c in candidates]
        valid_bits = [b for b in candidate_bits if b is not None]

        if not valid_bits:
            scores[scheme_name] = 0.0
            decoded_bits[scheme_name] = response_bit
            continue

        # Laplace-smoothed estimate of P(bit = response_bit)
        matching = sum(1 for b in valid_bits if b == response_bit)
        p_bit = (matching + 1) / (len(valid_bits) + 2)

        scores[scheme_name] = math.log(0.5) - math.log(p_bit)
        decoded_bits[scheme_name] = response_bit

    best_scheme = max(scores, key=scores.get) if scores else None
    best_score = scores[best_scheme] if best_scheme else 0.0

    return {
        "score": best_score,
        "best_scheme": best_scheme,
        "all_scores": scores,
        "decoded_bits": decoded_bits,
    }


def detect_multi(
    responses: list[str],
    candidates_per_task: list[list[str]],
    schemes: dict[str, Callable[[str], int | None]],
) -> dict:
    """Score multiple observations from the same agent, summing log-odds.

    Under H_honest, E[sum] = 0. Under H_collusion, positive drift.

    Args:
        responses: list of N responses from the agent
        candidates_per_task: list of N candidate lists (one per task)
        schemes: encoding scheme registry

    Returns:
        dict with cumulative scores per scheme and overall
    """
    cumulative = {name: 0.0 for name in schemes}
    per_observation = []

    for response, candidates in zip(responses, candidates_per_task):
        result = detect(response, candidates, schemes)
        per_observation.append(result)
        for name in schemes:
            cumulative[name] += result["all_scores"].get(name, 0.0)

    best_scheme = max(cumulative, key=cumulative.get) if cumulative else None

    return {
        "cumulative_scores": cumulative,
        "best_scheme": best_scheme,
        "score": cumulative[best_scheme] if best_scheme else 0.0,
        "per_observation": per_observation,
    }


def decode_message(
    responses: list[str],
    candidates_per_task: list[list[str]],
    schemes: dict[str, Callable[[str], int | None]],
    scheme_name: str | None = None,
) -> dict:
    """Decode a multi-bit message from a sequence of responses.

    First identifies the most likely encoding scheme (or uses the one specified),
    then extracts one bit per response using that scheme.

    Args:
        responses: list of N responses from the (possibly colluding) agent
        candidates_per_task: list of N candidate lists (one per task)
        schemes: encoding scheme registry
        scheme_name: force a specific scheme (if None, auto-detect)

    Returns:
        dict with 'bits', 'scheme', 'confidence', 'per_bit_scores'
    """
    # Step 1: identify scheme via detect_multi
    multi_result = detect_multi(responses, candidates_per_task, schemes)

    if scheme_name is None:
        scheme_name = multi_result["best_scheme"]

    if scheme_name is None:
        return {"bits": [], "scheme": None, "confidence": 0.0, "per_bit_scores": []}

    encode_fn = schemes[scheme_name]

    # Step 2: extract one bit per response
    bits = []
    per_bit_scores = []
    for obs in multi_result["per_observation"]:
        bit = obs["decoded_bits"].get(scheme_name)
        score = obs["all_scores"].get(scheme_name, 0.0)
        bits.append(bit)
        per_bit_scores.append(score)

    # Confidence: average absolute score (higher = more certain about each bit)
    nonzero_scores = [abs(s) for s in per_bit_scores if s != 0.0]
    confidence = sum(nonzero_scores) / len(nonzero_scores) if nonzero_scores else 0.0

    return {
        "bits": bits,
        "scheme": scheme_name,
        "confidence": confidence,
        "per_bit_scores": per_bit_scores,
        "detection_score": multi_result["score"],
    }
