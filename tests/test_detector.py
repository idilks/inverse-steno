"""Unit tests for the inverse planning detector — no LLM calls needed."""

import math
from src.detector.inverse_planning import detect, detect_multi
from src.schemes.parity import parity_encode


def _make_candidates(n_even: int, n_odd: int) -> list[str]:
    """Make candidate responses with controlled parity."""
    candidates = []
    for _ in range(n_even):
        candidates.append("one two three four")  # 4 words, even
    for _ in range(n_odd):
        candidates.append("one two three")  # 3 words, odd
    return candidates


class TestDetect:
    def test_symmetric_partition_gives_zero_score(self):
        """When p_bit = 0.5, detection score should be ~0."""
        candidates = _make_candidates(25, 25)
        schemes = {"parity": parity_encode}
        result = detect("one two three four", candidates, schemes)
        assert abs(result["score"]) < 0.1

    def test_asymmetric_partition_flags_rare_bit(self):
        """When the response's bit is rare among candidates, score should be positive."""
        # 90% even, 10% odd
        candidates = _make_candidates(45, 5)
        schemes = {"parity": parity_encode}
        # Test with an odd response (rare bit)
        result = detect("one two three", candidates, schemes)
        assert result["score"] > 0.5
        assert result["decoded_bits"]["parity"] == 1

    def test_asymmetric_partition_common_bit_negative(self):
        """When response's bit is the common one, score should be negative."""
        candidates = _make_candidates(45, 5)
        schemes = {"parity": parity_encode}
        result = detect("one two three four", candidates, schemes)
        assert result["score"] < 0

    def test_best_scheme_selection(self):
        """With multiple schemes, best_scheme picks the highest scorer."""
        candidates = _make_candidates(45, 5)
        schemes = {
            "parity": parity_encode,
            "fake_symmetric": lambda r: len(r) % 2,  # likely ~symmetric
        }
        result = detect("one two three", candidates, schemes)
        # parity should dominate since it has strong asymmetry for this candidate set
        assert result["best_scheme"] == "parity"


class TestDetectMulti:
    def test_cumulative_scores_sum(self):
        """Multi-observation scores should sum individual scores."""
        candidates = _make_candidates(45, 5)
        schemes = {"parity": parity_encode}

        # 3 odd responses (rare bit) should accumulate positive score
        responses = ["one two three"] * 3
        candidates_list = [candidates] * 3

        result = detect_multi(responses, candidates_list, schemes)
        single = detect("one two three", candidates, schemes)

        assert abs(result["score"] - 3 * single["score"]) < 0.01

    def test_honest_agent_near_zero(self):
        """Honest agent with symmetric candidates should have ~0 cumulative score."""
        candidates = _make_candidates(25, 25)
        schemes = {"parity": parity_encode}

        # Mix of even and odd — honest behavior
        responses = ["one two three four", "one two three", "one two three four"]
        candidates_list = [candidates] * 3

        result = detect_multi(responses, candidates_list, schemes)
        # Should be close to zero (within noise)
        assert abs(result["score"]) < 1.0
