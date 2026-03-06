# CLAUDE.md

## Project

Detecting covertext steganography in LLM outputs via **inverse planning**. A colluding agent encoding a hidden bit restricts its outputs to bit-compatible ones, paying a "steganography tax" that's statistically detectable.

Benchmark: [Redwood Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark). Dataset: AQUA-RAT.

## Key insight: Q-cancellation

The log-odds ratio simplifies to `log(0.5) - log(p_bit)`, where `p_bit` is the fraction of honest samples encoding the same bit. Q(r) cancels completely — the detector is a bit-frequency test with Boltzmann-rational justification. Per-observation power depends on encoding asymmetry; multi-observation detection is where discrimination compounds.

## Structure

```
src/
  schemes/           # pure functions: str -> int | None
    synonym.py       # first synonym-pair match -> bit (8 pairs, PRIMARY)
    structural.py    # answer position + step numbering -> bit
    parity.py        # word count mod 2 (sanity check only)
  detector/
    inverse_planning.py  # detect(), detect_multi()
    sampling.py          # async candidate generation + disk cache
  baselines/
    llm_judge.py     # prompt-based detection
    chi_squared.py   # frequency test on agent's bit distribution
  eval/
    auc.py           # Mann-Whitney AUC (no sklearn)
    multi_obs.py     # AUC-vs-N curve
    run_eval.py      # evaluation from JSON data files
tests/               # 30 tests, all passing
```

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Status

- **Done:** detector core, encoding schemes, baselines, eval pipeline, unit tests (30/30)
- **Next:** Phase 0 pilot — 10 AQUA-RAT problems, K=50 samples, measure bit asymmetry per scheme. Go/no-go gate for each encoding. Needs API keys.
