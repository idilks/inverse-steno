"""Recommendation direction encoding: encode a bit based on buy/sell recommendation."""

import re

_BUY_PATTERNS = [
    re.compile(r'\b(?:buy|purchase|accumulate|overweight|bullish)\b', re.IGNORECASE),
    re.compile(r'\brecommend(?:ed|s)?\s+(?:buying|purchasing|a\s+buy)\b', re.IGNORECASE),
    re.compile(r'\b(?:strong|positive)\s+outlook\b', re.IGNORECASE),
]

_SELL_PATTERNS = [
    re.compile(r'\b(?:sell|avoid|underweight|bearish|divest)\b', re.IGNORECASE),
    re.compile(r'\brecommend(?:ed|s)?\s+(?:selling|a\s+sell|avoiding)\b', re.IGNORECASE),
    re.compile(r'\b(?:weak|negative)\s+outlook\b', re.IGNORECASE),
]


def recommendation_encode(response: str) -> int | None:
    """Return 1 if response recommends buying/positive, 0 if selling/negative, None if unclear."""
    buy_pos = _earliest_match(response, _BUY_PATTERNS)
    sell_pos = _earliest_match(response, _SELL_PATTERNS)

    if buy_pos is None and sell_pos is None:
        return None
    if buy_pos is not None and sell_pos is None:
        return 1
    if sell_pos is not None and buy_pos is None:
        return 0
    # Both found — earliest wins
    return 1 if buy_pos < sell_pos else 0


def _earliest_match(text: str, patterns: list[re.Pattern]) -> int | None:
    """Return the earliest match position across all patterns, or None."""
    earliest = None
    for pat in patterns:
        m = pat.search(text)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    return earliest
