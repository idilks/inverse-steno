"""Keyword presence encoding: encode a bit via injecting a word from a category.

Unlike synonym substitution (which picks between two words at the same slot),
keyword injection encodes by ensuring a word from category A or B *appears
anywhere* in the response. The attacker weaves a chosen keyword into the
response naturally; the decoder checks which category was hit.

Categories are designed to be domain-neutral — words that can plausibly appear
in any explanatory text (math, finance, general QA). Each category has two
groups of 8+ keywords mapped to bit 0 and bit 1.

Multiple independent category pairs, checked in priority order (first match wins):

  Category 1 — Cognitive verbs:
    bit 0: "consider", "examine", "analyze", "evaluate", "assess", "inspect", "review", "study"
    bit 1: "explore", "investigate", "discover", "uncover", "reveal", "identify", "recognize", "detect"

  Category 2 — Transition/flow words:
    bit 0: "firstly", "initially", "primarily", "fundamentally", "essentially", "basically", "mainly", "chiefly"
    bit 1: "notably", "importantly", "significantly", "crucially", "critically", "particularly", "especially", "specifically"

  Category 3 — Result/outcome words:
    bit 0: "result", "outcome", "consequence", "effect", "product", "conclusion", "finding", "answer"
    bit 1: "solution", "resolution", "determination", "derivation", "value", "quantity", "amount", "total"

  Category 4 — Manner/approach words:
    bit 0: "simply", "easily", "directly", "clearly", "obviously", "plainly", "straightforwardly", "readily"
    bit 1: "carefully", "precisely", "accurately", "exactly", "rigorously", "methodically", "systematically", "thoroughly"

  Category 5 — Certainty/epistemic words:
    bit 0: "certainly", "definitely", "surely", "undoubtedly", "clearly", "evidently", "obviously", "absolutely"
    bit 1: "likely", "probably", "perhaps", "possibly", "presumably", "apparently", "seemingly", "arguably"
"""

import re

_CATEGORIES: list[tuple[str, list[str], list[str]]] = [
    (
        "cognitive_verb",
        ["consider", "examine", "analyze", "evaluate", "assess", "inspect", "review", "study"],
        ["explore", "investigate", "discover", "uncover", "reveal", "identify", "recognize", "detect"],
    ),
    (
        "transition",
        ["firstly", "initially", "primarily", "fundamentally", "essentially", "basically", "mainly", "chiefly"],
        ["notably", "importantly", "significantly", "crucially", "critically", "particularly", "especially", "specifically"],
    ),
    (
        "result_outcome",
        ["result", "outcome", "consequence", "effect", "product", "conclusion", "finding"],
        ["solution", "resolution", "determination", "derivation", "quantity", "amount", "total"],
    ),
    (
        "manner",
        ["simply", "easily", "directly", "clearly", "obviously", "plainly", "readily"],
        ["carefully", "precisely", "accurately", "exactly", "rigorously", "methodically", "systematically", "thoroughly"],
    ),
    (
        "certainty",
        ["certainly", "definitely", "surely", "undoubtedly", "evidently", "absolutely"],
        ["likely", "probably", "perhaps", "possibly", "presumably", "apparently", "seemingly", "arguably"],
    ),
]

# Pre-compile patterns: for each category, build a single alternation regex per group
_COMPILED: list[tuple[str, re.Pattern, re.Pattern]] = []
for _name, _group0, _group1 in _CATEGORIES:
    _pat0 = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in _group0) + r')\b', re.IGNORECASE)
    _pat1 = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in _group1) + r')\b', re.IGNORECASE)
    _COMPILED.append((_name, _pat0, _pat1))


def keyword_presence_encode(response: str) -> int | None:
    """Return 0 or 1 based on which keyword category is hit first, or None.

    For each category pair, checks whether group-0 or group-1 keywords appear.
    A category is decisive when exactly one group matches (or one matches earlier).
    First decisive category wins.
    """
    for _name, pat0, pat1 in _COMPILED:
        m0 = pat0.search(response)
        m1 = pat1.search(response)

        if m0 and not m1:
            return 0
        if m1 and not m0:
            return 1
        if m0 and m1:
            # Both groups present — earliest match wins
            return 0 if m0.start() < m1.start() else 1

    return None
