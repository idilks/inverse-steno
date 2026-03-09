# Notes: Redwood-Style 4-Bit Attacks

## How Redwood encodes 4 bits per response

Each attacker has a **family of 16 variants** (2^4 = 16). The 4-bit message selects which variant to use. One response, one scheme family, 4 bits.

| Attacker | How it encodes | What the 16 variants are |
|---|---|---|
| IntroInjection | Forces the LLM to start with a specific phrase | 16 intro phrases ("Let me solve this step by step.", "I'll work through this problem carefully.", etc.) |
| ToneModifier | Prompt includes a tone instruction | 16 tone instructions ("formal academic", "casual friendly", etc.) |
| Scratchpad | Embeds a 4-char binary string ("0110") in the response | 16 possible 4-bit strings |

These are all Tier 1 (prompt-based, no distribution preservation). The LLM is explicitly told what to do.

## How to test with our detector

### The problem

Our detector uses **binary** partition functions (response → 0 or 1). A 4-bit scheme has 16 categories. We need to either:

1. **Decompose into 4 binary partitions.** E.g., for IntroInjection:
   - Bit 0: phrase index & 8 (phrases 0-7 vs 8-15)
   - Bit 1: phrase index & 4
   - Bit 2: phrase index & 2
   - Bit 3: phrase index & 1
   Each binary partition is independently testable with our existing detector.

2. **Use the existing partition library.** Many of our partitions already correlate with these encoding families:
   - IntroInjection: `starts_with_verb`, `first_word_vowel`, `first_letter_half` might partially capture phrase selection
   - ToneModifier: `has_exclamation`, `word_count_short`, `has_step_numbering`, `first_sentence_short` correlate with tone
   - Scratchpad: `high_digit_density` might catch the embedded binary string

   This is the codebook-free case — we don't know the exact scheme, but our library has correlated features.

### What we need

**To generate Redwood-style stego data:**
```python
from src.benchmark.attackers import ATTACKERS
from scripts.bridge_eval import make_model_fn

model_fn = make_model_fn("anthropic.claude-haiku-4-5-20251001")
attacker = ATTACKERS["intro_injection"]

# For each AQUA-RAT question, generate one stego response with random 4-bit payload
import random
bits = [random.choice([True, False]) for _ in range(4)]
prompt = attacker.encode_prompt(question_text, bits)
response = await model_fn(prompt, 1)  # single call
```

**To evaluate:**
1. Generate stego responses for 100 AQUA-RAT questions (100 API calls per attacker × 3 attackers = 300 calls)
2. Also generate 100 honest responses (100 calls, or sample from existing candidate cache)
3. Run `detect()` with our partition library against existing cached reference candidates
4. Compute AUC

### API cost estimate

| Item | Calls |
|---|---|
| Stego responses (3 attackers × 100 questions) | 300 |
| Honest responses (already cached) | 0 |
| Reference candidates (already cached, 100 per question) | 0 |
| **Total new calls** | **300** |

### Script outline

```bash
# Step 1: Generate stego responses
python -m scripts.generate_redwood_stego --n-questions 100 --attackers intro,tone,scratchpad

# Step 2: Evaluate (uses existing cached references)
python -m scripts.eval_aqua_redwood --experiment all --dry-run
```

### Expected results

- IntroInjection: easiest to detect (forced prefix is highly unnatural, p_bit for `first_word_vowel` or similar will be extreme)
- ToneModifier: moderate (tone affects many features, but individual partitions may only weakly correlate)
- Scratchpad: easiest for a judge, but hardest for partition-based detection (the "0110" string doesn't map to any of our partition features)

### Key question for the paper

Does our codebook-free detector catch Redwood-style attacks that it was never trained on, using only correlated partition features? If yes: strong generalization claim. If no for scratchpad but yes for intro/tone: interesting — shows which attack families are covered by surface-level partitions vs which need deeper features.

## How this differs from our current B=1 experiments

| | Current (B=1) | Redwood (B=4) |
|---|---|---|
| Bits per response | 1 | 4 |
| How encoding works | Binary partition (synonym/structural/parity) | 16-category selection |
| Attacker constraint | Must land in 1/2 of response space | Must land in 1/16 of response space |
| Detection difficulty | Baseline | Should be easier (larger constraint) |
| What we need | Already have data | Need 300 API calls |

## Relationship to mixed-scheme attacks

If an attacker uses DIFFERENT schemes across responses (synonym for Q1, structural for Q2), our sum-then-max still works but with reduced power — each partition only accumulates signal from a fraction of observations. This is a real evasion strategy but has costs:
- Attacker needs the receiver to know the schedule (which scheme for which question)
- Each partition gets N/K observations instead of N (K schemes used)
- Detection power degrades gracefully, doesn't collapse
