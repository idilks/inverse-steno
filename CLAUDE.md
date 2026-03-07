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
    synonym.py       # first synonym-pair match -> bit (16 pairs, PRIMARY)
    structural.py    # answer position + step numbering -> bit
    parity.py        # word count mod 2 (sanity check only)
    recommendation.py # buy/sell signal -> bit
  detector/
    inverse_planning.py  # detect(), detect_multi(), decode_message()
    sampling.py          # async candidate generation + disk cache
  baselines/
    llm_judge.py     # prompt-based detection
    chi_squared.py   # frequency test on agent's bit distribution
  eval/
    auc.py           # Mann-Whitney AUC (no sklearn)
    multi_obs.py     # AUC-vs-N curve
    run_eval.py      # evaluation from JSON data files + benchmark results
  benchmark/
    attackers.py     # 3 Redwood-style attackers (intro, scratchpad, tone), 4 bits each
    dataset.py       # AQUA-RAT loader (HuggingFace + local cache)
    run_benchmark.py # full pipeline: both conditions, AUC + BER metrics
tests/               # 70 tests, all passing
```

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Status

- **Done:** detector core, encoding schemes, baselines, eval pipeline, benchmark integration (attackers + dataset + pipeline), decode_message(), unit tests (70/70)
- **Next:** Phase 0 pilot — run benchmark with API keys. `run_full_benchmark(model_fn=..., n_questions=10, k_candidates=50)`. Needs Gemini 2.5 Flash access + `datasets` pip package (or cached aqua_rat.json).
