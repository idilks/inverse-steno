"""Unit tests for encoding schemes."""

import pytest
from src.schemes.synonym import synonym_encode
from src.schemes.structural import structural_encode
from src.schemes.parity import parity_encode
from src.schemes.recommendation import recommendation_encode


class TestSynonymEncode:
    def test_therefore_is_bit_0(self):
        assert synonym_encode("We add 3 + 4, therefore the answer is 7.") == 0

    def test_thus_is_bit_1(self):
        assert synonym_encode("We add 3 + 4, thus the answer is 7.") == 1

    def test_calculate_is_bit_0(self):
        assert synonym_encode("We need to calculate the total.") == 0

    def test_compute_is_bit_1(self):
        assert synonym_encode("We need to compute the total.") == 1

    def test_first_match_wins(self):
        # "thus" appears before "calculate"
        assert synonym_encode("Thus we calculate the result.") == 1

    def test_case_insensitive(self):
        assert synonym_encode("THEREFORE the answer is clear.") == 0

    def test_no_match_returns_none(self):
        assert synonym_encode("The answer is 42.") is None

    def test_is_equal_to(self):
        assert synonym_encode("x is equal to 5") == 1

    def test_equals(self):
        assert synonym_encode("x equals 5") == 0

    def test_plug_in(self):
        assert synonym_encode("Now plug in x = 3") == 1

    def test_we_have_vs_we_get(self):
        assert synonym_encode("After simplifying, we have 10.") == 0
        assert synonym_encode("After simplifying, we get 10.") == 1

    # General pairs
    def test_however_vs_but(self):
        assert synonym_encode("However, the data shows otherwise.") == 0
        assert synonym_encode("But the data shows otherwise.") == 1

    def test_additionally_vs_also(self):
        assert synonym_encode("Additionally, revenue grew 10%.") == 0
        assert synonym_encode("Also, revenue grew 10%.") == 1

    def test_for_example_vs_for_instance(self):
        assert synonym_encode("For example, consider ACME Corp.") == 0
        assert synonym_encode("For instance, consider ACME Corp.") == 1

    def test_recommend_vs_suggest(self):
        assert synonym_encode("I recommend buying the stock.") == 0
        assert synonym_encode("I suggest buying the stock.") == 1

    # Financial pairs
    def test_increase_vs_rise(self):
        assert synonym_encode("Margins saw an increase of 12%.") == 0
        assert synonym_encode("Margins saw a rise of 12%.") == 1

    def test_decline_vs_drop(self):
        assert synonym_encode("Margins saw a decline this quarter.") == 0
        assert synonym_encode("Margins saw a drop this quarter.") == 1

    def test_revenue_vs_earnings(self):
        assert synonym_encode("The revenue exceeded expectations.") == 0
        assert synonym_encode("The earnings exceeded expectations.") == 1


class TestStructuralEncode:
    def test_answer_last(self):
        response = (
            "First we add 3 + 4 = 7. Then we multiply by 2 to get 14. "
            "We subtract 1 to get 13. After all these steps, "
            "we can see that the answer is B."
        )
        assert structural_encode(response) == 0

    def test_answer_first(self):
        response = (
            "The answer is B. "
            "To see why, we add 3 + 4 = 7, then multiply by 2 to get 14, "
            "then subtract 1 to get 13, which matches option B."
        )
        assert structural_encode(response) == 1

    def test_numbered_steps(self):
        response = (
            "Step 1: Add 3 + 4 = 7\n"
            "Step 2: Multiply by 2 = 14\n"
            "Step 3: Subtract 1 = 13"
        )
        assert structural_encode(response) == 0

    def test_prose_flow(self):
        response = (
            "First we add 3 and 4 to get 7, then multiply by 2 "
            "for 14, and finally subtract 1 to arrive at 13."
        )
        assert structural_encode(response) == 1

    def test_empty_returns_none(self):
        assert structural_encode("") is None
        assert structural_encode("   ") is None


class TestParityEncode:
    def test_even_word_count(self):
        assert parity_encode("one two three four") == 0

    def test_odd_word_count(self):
        assert parity_encode("one two three") == 1

    def test_single_word(self):
        assert parity_encode("hello") == 1

    def test_empty(self):
        assert parity_encode("") == 0


class TestRecommendationEncode:
    def test_buy_returns_1(self):
        assert recommendation_encode("I would buy this stock given the strong fundamentals.") == 1

    def test_sell_returns_0(self):
        assert recommendation_encode("Investors should sell before further decline.") == 0

    def test_bullish_returns_1(self):
        assert recommendation_encode("The outlook is bullish for next quarter.") == 1

    def test_bearish_returns_0(self):
        assert recommendation_encode("Market conditions remain bearish.") == 0

    def test_avoid_returns_0(self):
        assert recommendation_encode("I would avoid this stock at current levels.") == 0

    def test_positive_outlook_returns_1(self):
        assert recommendation_encode("Management projects a positive outlook for FY25.") == 1

    def test_no_signal_returns_none(self):
        assert recommendation_encode("The company reported Q3 results today.") is None

    def test_earliest_wins_buy_first(self):
        assert recommendation_encode("Buy now before you need to sell later.") == 1

    def test_earliest_wins_sell_first(self):
        assert recommendation_encode("Sell the old position, then buy the dip.") == 0

    def test_case_insensitive(self):
        assert recommendation_encode("STRONGLY RECOMMEND BUYING this stock.") == 1
