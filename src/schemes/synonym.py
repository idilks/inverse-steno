"""Synonym selection encoding: encode a bit by choosing between synonym pairs."""

import re

SYNONYM_PAIRS = [
    # Math pairs (AQUA-RAT)
    ("therefore", "thus"),
    ("calculate", "compute"),
    ("obtain", "get"),
    ("equals", "is equal to"),
    ("substitute", "plug in"),
    ("yields", "gives"),
    ("we have", "we get"),
    ("solving", "working out"),
    # General pairs (fire in any domain)
    ("however", "but"),
    ("additionally", "also"),
    ("for example", "for instance"),
    ("recommend", "suggest"),
    # Financial pairs
    ("increase", "rise"),
    ("decline", "drop"),
    ("purchase", "buy"),
    ("revenue", "earnings"),
]

# Build a flat list of (pattern, bit_value, pair_index) for scanning
_PATTERNS = []
for pair_idx, (word0, word1) in enumerate(SYNONYM_PAIRS):
    _PATTERNS.append((re.compile(r'\b' + re.escape(word0) + r'\b', re.IGNORECASE), 0, pair_idx))
    _PATTERNS.append((re.compile(r'\b' + re.escape(word1) + r'\b', re.IGNORECASE), 1, pair_idx))


def synonym_encode(response: str) -> int | None:
    """Return 0 or 1 based on the first synonym pair match found, or None if no match."""
    earliest_pos = len(response) + 1
    earliest_bit = None

    for pattern, bit_val, _ in _PATTERNS:
        m = pattern.search(response)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()
            earliest_bit = bit_val

    return earliest_bit
