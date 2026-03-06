"""Tests for benchmark module — attackers, dataset, and decode_message."""

import pytest
from src.benchmark.attackers import (
    IntroInjectionAttacker,
    ScratchpadAttacker,
    ToneModifierAttacker,
    _bits_to_index,
    _index_to_bits,
    INTRO_PHRASES,
    TONE_INSTRUCTIONS,
)
from src.benchmark.dataset import format_question
from src.detector.inverse_planning import decode_message
from src.schemes.parity import parity_encode


class TestBitConversion:
    def test_roundtrip_all_false(self):
        bits = [False, False, False, False]
        assert _index_to_bits(_bits_to_index(bits, 4), 4) == bits

    def test_roundtrip_all_true(self):
        bits = [True, True, True, True]
        assert _index_to_bits(_bits_to_index(bits, 4), 4) == bits

    def test_roundtrip_mixed(self):
        bits = [True, False, True, False]
        assert _index_to_bits(_bits_to_index(bits, 4), 4) == bits

    def test_index_range(self):
        for idx in range(16):
            bits = _index_to_bits(idx, 4)
            assert _bits_to_index(bits, 4) == idx


class TestIntroInjectionAttacker:
    def setup_method(self):
        self.attacker = IntroInjectionAttacker()

    def test_bits_per_response(self):
        assert self.attacker.bits_per_response == 4

    def test_encode_prompt_contains_intro(self):
        bits = [False, False, False, False]  # index 0
        prompt = self.attacker.encode_prompt("What is 2+2?", bits)
        assert INTRO_PHRASES[0] in prompt

    def test_encode_prompt_different_bits_different_intros(self):
        p0 = self.attacker.encode_prompt("Q?", [False, False, False, False])
        p1 = self.attacker.encode_prompt("Q?", [True, False, False, False])
        assert p0 != p1

    def test_decode_matches_intro(self):
        for idx in range(16):
            bits = _index_to_bits(idx, 4)
            answer = INTRO_PHRASES[idx] + " The answer is 4."
            decoded = self.attacker.decode("Q?", answer)
            assert decoded == bits, f"Failed for idx={idx}"

    def test_decode_no_match_returns_zeros(self):
        decoded = self.attacker.decode("Q?", "Random text with no intro.")
        assert decoded == [False] * 4


class TestScratchpadAttacker:
    def setup_method(self):
        self.attacker = ScratchpadAttacker()

    def test_bits_per_response(self):
        assert self.attacker.bits_per_response == 4

    def test_encode_prompt_contains_marker(self):
        bits = [True, False, True, True]
        prompt = self.attacker.encode_prompt("Q?", bits)
        assert "1011" in prompt

    def test_decode_finds_marker(self):
        bits = [True, False, True, True]
        answer = "In step 1011, we calculate the answer is 5."
        decoded = self.attacker.decode("Q?", answer)
        assert decoded == bits

    def test_decode_no_marker_returns_zeros(self):
        answer = "The answer is simply 42."
        decoded = self.attacker.decode("Q?", answer)
        assert decoded == [False] * 4


class TestToneModifierAttacker:
    def setup_method(self):
        self.attacker = ToneModifierAttacker()

    def test_bits_per_response(self):
        assert self.attacker.bits_per_response == 4

    def test_encode_prompt_contains_tone(self):
        bits = [False, False, False, False]  # index 0
        prompt = self.attacker.encode_prompt("Q?", bits)
        assert TONE_INSTRUCTIONS[0] in prompt

    def test_decode_with_step_numbers(self):
        answer = "Step 1: Calculate. Step 2: Simplify. The answer is 7."
        decoded = self.attacker.decode("Q?", answer)
        # Should detect step numbering -> index 8
        assert decoded == _index_to_bits(8, 4)

    def test_decode_enthusiastic(self):
        # Long answer with exclamation marks -> enthusiastic (index 4)
        # Must be > 50 words to avoid triggering concise/minimal heuristics
        answer = (
            "Wow! This is such a great question! I am really excited to work through "
            "this problem! The key insight here is that we need to carefully consider "
            "each option! After thorough analysis of all the given information, I can "
            "confidently say that the answer is definitely 42! What a wonderful result! "
            "This is a beautiful example of mathematical reasoning in action! Amazing!"
        )
        decoded = self.attacker.decode("Q?", answer)
        assert decoded == _index_to_bits(4, 4)


class TestFormatQuestion:
    def test_format_with_list_choices(self):
        item = {
            "task_id": "test_0",
            "question": "What is 2+2?",
            "choices": ["A) 3", "B) 4", "C) 5"],
            "answer": "B",
        }
        result = format_question(item)
        assert "What is 2+2?" in result
        assert "A) 3" in result
        assert "B) 4" in result

    def test_format_with_string_choices(self):
        item = {
            "task_id": "test_0",
            "question": "What is 2+2?",
            "choices": "A) 3\nB) 4\nC) 5",
            "answer": "B",
        }
        result = format_question(item)
        assert "What is 2+2?" in result
        assert "A) 3" in result


class TestDecodeMessage:
    def test_decode_uniform_bits(self):
        """All responses encode the same bit -> that bit is decoded."""
        # All odd-word responses
        responses = ["one two three"] * 5
        candidates = ["one two three four"] * 45 + ["one two three"] * 5
        candidates_per_task = [candidates] * 5
        schemes = {"parity": parity_encode}

        result = decode_message(responses, candidates_per_task, schemes)
        assert result["scheme"] == "parity"
        assert all(b == 1 for b in result["bits"])
        assert result["confidence"] > 0

    def test_decode_mixed_bits(self):
        """Mixed responses -> mixed decoded bits."""
        responses = [
            "one two three four",  # even = 0
            "one two three",       # odd = 1
            "one two three four",  # even = 0
        ]
        candidates = ["one two three four"] * 25 + ["one two three"] * 25
        candidates_per_task = [candidates] * 3
        schemes = {"parity": parity_encode}

        result = decode_message(responses, candidates_per_task, schemes)
        assert result["bits"] == [0, 1, 0]

    def test_decode_with_forced_scheme(self):
        """Forcing a scheme name works."""
        responses = ["one two three"] * 3
        candidates = ["one two three four"] * 25 + ["one two three"] * 25
        candidates_per_task = [candidates] * 3
        schemes = {"parity": parity_encode}

        result = decode_message(
            responses, candidates_per_task, schemes, scheme_name="parity"
        )
        assert result["scheme"] == "parity"

    def test_decode_empty(self):
        result = decode_message([], [], {"parity": parity_encode})
        assert result["bits"] == []
