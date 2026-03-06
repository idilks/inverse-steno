"""Structural choice encoding: encode a bit based on response organization."""

import re


_ANSWER_PATTERN = re.compile(
    r'(?:the answer is|answer:\s*)\s*\(?[A-E]\)?',
    re.IGNORECASE,
)

_STEP_PATTERN = re.compile(r'step\s+\d', re.IGNORECASE)


def structural_encode(response: str) -> int | None:
    """Return bit based on response structure.

    Checks two independent signals and combines them:
    1. Answer position: answer in first 30% of text -> 1, last 30% -> 0
    2. Numbered steps present -> 0, prose flow -> 1

    Uses whichever signal fires first. Returns None if neither is detectable.
    """
    text = response.strip()
    if not text:
        return None

    answer_match = _ANSWER_PATTERN.search(text)
    has_steps = bool(_STEP_PATTERN.search(text))

    # Signal 1: answer position
    answer_bit = None
    if answer_match:
        relative_pos = answer_match.start() / len(text)
        if relative_pos < 0.3:
            answer_bit = 1  # answer-first
        elif relative_pos > 0.7:
            answer_bit = 0  # answer-last (work-first)

    # Signal 2: numbered steps
    step_bit = 0 if has_steps else 1

    # Prefer answer position signal if available (more discriminative)
    if answer_bit is not None:
        return answer_bit
    return step_bit
