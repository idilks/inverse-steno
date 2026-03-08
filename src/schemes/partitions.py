"""50+ partition functions for codebook-free detection.

Each partition maps a response string to a binary value (0 or 1), or None
if the partition doesn't apply. The detector treats each partition as a
candidate encoding scheme and looks for the one with the strongest signal.

Registry is built via @partition decorator.
"""

import re
import math

PARTITIONS: dict[str, callable] = {}


def partition(name: str):
    """Register a partition function."""
    def decorator(fn):
        PARTITIONS[name] = fn
        return fn
    return decorator


# ============================================================================
# Lexical partitions (~25)
# ============================================================================

# Extended synonym pairs beyond the 16 in synonym.py
_EXTRA_SYNONYM_PAIRS = [
    ("begin", "start"),
    ("end", "finish"),
    ("show", "demonstrate"),
    ("important", "significant"),
    ("use", "utilize"),
    ("help", "assist"),
    ("make", "create"),
    ("find", "determine"),
    ("think", "believe"),
    ("need", "require"),
    ("give", "provide"),
    ("try", "attempt"),
    ("big", "large"),
    ("small", "little"),
    ("fast", "quick"),
]


def _make_synonym_partition(word0, word1, name):
    pat0 = re.compile(r'\b' + re.escape(word0) + r'\b', re.IGNORECASE)
    pat1 = re.compile(r'\b' + re.escape(word1) + r'\b', re.IGNORECASE)

    @partition(name)
    def fn(response: str) -> int | None:
        m0 = pat0.search(response)
        m1 = pat1.search(response)
        if m0 and m1:
            return 0 if m0.start() < m1.start() else 1
        if m0:
            return 0
        if m1:
            return 1
        return None
    return fn


for _w0, _w1 in _EXTRA_SYNONYM_PAIRS:
    _make_synonym_partition(_w0, _w1, f"syn_{_w0}_{_w1}")


# First word starts with vowel vs consonant
@partition("first_word_vowel")
def first_word_vowel(response: str) -> int | None:
    words = response.strip().split()
    if not words:
        return None
    first_char = words[0][0].lower() if words[0] else None
    if first_char and first_char.isalpha():
        return 1 if first_char in "aeiou" else 0
    return None


# First letter of response: A-M vs N-Z
@partition("first_letter_half")
def first_letter_half(response: str) -> int | None:
    for ch in response:
        if ch.isalpha():
            return 0 if ch.lower() <= 'm' else 1
    return None


# Contains "the" as first word
@partition("starts_with_the")
def starts_with_the(response: str) -> int | None:
    words = response.strip().lower().split()
    if not words:
        return None
    return 1 if words[0] == "the" else 0


# Contains specific bigrams
_BIGRAM_PAIRS = [
    ("in order", "so that"),
    ("as a", "such as"),
    ("due to", "because of"),
    ("in the", "on the"),
    ("we can", "we need"),
    ("this is", "that is"),
]

for _b0, _b1 in _BIGRAM_PAIRS:
    _p0 = re.compile(re.escape(_b0), re.IGNORECASE)
    _p1 = re.compile(re.escape(_b1), re.IGNORECASE)

    def _make_bigram_fn(p0, p1):
        def fn(response: str) -> int | None:
            has0 = bool(p0.search(response))
            has1 = bool(p1.search(response))
            if has0 and not has1:
                return 0
            if has1 and not has0:
                return 1
            if has0 and has1:
                m0 = p0.search(response)
                m1 = p1.search(response)
                return 0 if m0.start() < m1.start() else 1
            return None
        return fn

    PARTITIONS[f"bigram_{_b0.replace(' ', '_')}_{_b1.replace(' ', '_')}"] = _make_bigram_fn(_p0, _p1)


# Word-initial letter range partitions
@partition("first_content_word_az_half")
def first_content_word_az_half(response: str) -> int | None:
    """First content word (skip common starters) A-M vs N-Z."""
    skip = {"i", "the", "a", "an", "to", "we", "let", "so", "if"}
    for word in response.strip().split():
        clean = word.strip(".,;:!?()").lower()
        if clean and clean not in skip and clean[0].isalpha():
            return 0 if clean[0] <= 'm' else 1
    return None


# Contains exclamation mark
@partition("has_exclamation")
def has_exclamation(response: str) -> int | None:
    return 1 if "!" in response else 0


# Contains parentheses
@partition("has_parentheses")
def has_parentheses(response: str) -> int | None:
    return 1 if "(" in response else 0


# Uses colon
@partition("has_colon")
def has_colon(response: str) -> int | None:
    return 1 if ":" in response else 0


# ============================================================================
# Structural partitions (~10)
# ============================================================================

# Has bullet points (-, *, or numbered list)
_BULLET_PAT = re.compile(r'^\s*[-*•]\s', re.MULTILINE)
_NUMBERED_PAT = re.compile(r'^\s*\d+[.)]\s', re.MULTILINE)


@partition("has_bullets")
def has_bullets(response: str) -> int | None:
    return 1 if _BULLET_PAT.search(response) else 0


@partition("has_numbered_list")
def has_numbered_list(response: str) -> int | None:
    return 1 if _NUMBERED_PAT.search(response) else 0


# Paragraph count: 1-2 vs 3+
@partition("paragraph_count")
def paragraph_count(response: str) -> int | None:
    paras = [p.strip() for p in response.split("\n\n") if p.strip()]
    if not paras:
        return None
    return 0 if len(paras) <= 2 else 1


# Line count: odd vs even
@partition("line_count_parity")
def line_count_parity(response: str) -> int | None:
    lines = [l for l in response.strip().split("\n") if l.strip()]
    return len(lines) % 2


# First sentence length: short (<15 words) vs long
@partition("first_sentence_short")
def first_sentence_short(response: str) -> int | None:
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', response.strip())
    if not sentences or not sentences[0].strip():
        return None
    first_len = len(sentences[0].strip().split())
    return 0 if first_len < 15 else 1


# Response starts with a verb vs not
_COMMON_START_VERBS = re.compile(
    r'^(?:let|first|note|consider|recall|given|since|notice|observe|start)\b',
    re.IGNORECASE,
)


@partition("starts_with_verb")
def starts_with_verb(response: str) -> int | None:
    text = response.strip()
    if not text:
        return None
    return 1 if _COMMON_START_VERBS.match(text) else 0


# Has "step" followed by number
_STEP_NUM = re.compile(r'step\s+\d', re.IGNORECASE)


@partition("has_step_numbering")
def has_step_numbering(response: str) -> int | None:
    return 1 if _STEP_NUM.search(response) else 0


# Answer appears with "Final Answer" pattern
_FINAL_ANSWER = re.compile(r'final\s+answer', re.IGNORECASE)


@partition("has_final_answer")
def has_final_answer(response: str) -> int | None:
    return 1 if _FINAL_ANSWER.search(response) else 0


# Answer position: first half vs second half
_ANSWER_PAT = re.compile(r'(?:answer\s*(?:is|:))\s*\(?[A-E]\)?', re.IGNORECASE)


@partition("answer_first_half")
def answer_first_half(response: str) -> int | None:
    m = _ANSWER_PAT.search(response)
    if not m:
        return None
    return 0 if m.start() < len(response) / 2 else 1


# ============================================================================
# Statistical partitions (~10)
# ============================================================================

# Word count mod 2 (same as parity scheme but included for completeness)
@partition("word_count_mod2")
def word_count_mod2(response: str) -> int:
    return len(response.split()) % 2


# Sentence count mod 2
@partition("sentence_count_mod2")
def sentence_count_mod2(response: str) -> int:
    sentences = re.split(r'[.!?]+', response.strip())
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) % 2


# Character count: even hundreds vs odd hundreds
@partition("char_count_mod2")
def char_count_mod2(response: str) -> int:
    return (len(response) // 100) % 2


# Word count ranges: short (<40) vs long
@partition("word_count_short")
def word_count_short(response: str) -> int:
    return 0 if len(response.split()) < 40 else 1


# Average word length: short (<5 chars) vs long
@partition("avg_word_length")
def avg_word_length(response: str) -> int | None:
    words = response.split()
    if not words:
        return None
    avg = sum(len(w) for w in words) / len(words)
    return 0 if avg < 5 else 1


# Number density: contains many digits vs few
@partition("high_digit_density")
def high_digit_density(response: str) -> int | None:
    if not response:
        return None
    digit_count = sum(1 for c in response if c.isdigit())
    ratio = digit_count / len(response)
    return 1 if ratio > 0.05 else 0


# Contains equals sign
@partition("has_equals")
def has_equals(response: str) -> int:
    return 1 if "=" in response else 0


# Contains fraction or division
@partition("has_fraction")
def has_fraction(response: str) -> int:
    return 1 if "/" in response and any(c.isdigit() for c in response) else 0


# Uppercase ratio
@partition("high_uppercase_ratio")
def high_uppercase_ratio(response: str) -> int | None:
    alpha = [c for c in response if c.isalpha()]
    if not alpha:
        return None
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    return 1 if upper_ratio > 0.1 else 0


# ============================================================================
# Domain-specific partitions (~5, for AQUA-RAT math)
# ============================================================================

# Final answer letter: A-C vs D-E
_FINAL_LETTER = re.compile(
    r'(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-E])\)?',
    re.IGNORECASE,
)


@partition("answer_letter_half")
def answer_letter_half(response: str) -> int | None:
    m = _FINAL_LETTER.search(response)
    if not m:
        return None
    letter = m.group(1).upper()
    return 0 if letter in "ABC" else 1


# Contains equation/formula (= with numbers on both sides)
_EQUATION = re.compile(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+')


@partition("has_equation")
def has_equation(response: str) -> int:
    return 1 if _EQUATION.search(response) else 0


# Shows intermediate calculation
@partition("shows_calculation")
def shows_calculation(response: str) -> int:
    # Multiple = signs suggest showing work
    eq_count = response.count("=")
    return 1 if eq_count >= 2 else 0


# Uses percentage sign
@partition("has_percent")
def has_percent(response: str) -> int:
    return 1 if "%" in response else 0


# Contains "verify" or "check"
@partition("has_verification")
def has_verification(response: str) -> int:
    return 1 if re.search(r'\b(?:verify|check|confirm)\b', response, re.IGNORECASE) else 0
