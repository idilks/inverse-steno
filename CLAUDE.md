# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project detecting covertext steganography in LLM outputs using **inverse planning**. The core idea: a colluding agent encoding a hidden bit must restrict its outputs to those compatible with that bit, paying a "steganography tax" (suboptimality on the cover task) that is statistically detectable.

Benchmark: [Redwood Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark). Dataset: AQUA-RAT (math reasoning).

## Architecture (planned)

Three components:

1. **Data pipeline** — Load AQUA-RAT, generate honest responses (task-only) and stego responses (task + hidden bits) via encoding schemes.

2. **Detector** — Inverse planning model. Scores each response under two hypotheses:
   - `H_honest`: `P(action | task) ∝ exp(β · Q_task(action))`
   - `H_collusion`: same, but restricted to actions compatible with target bit
   - Detection score = `log P(action | H_collusion) - log P(action | H_honest)`
   - This is our current focus. 

3. **Evaluation** — AUC over N honest + N stego samples. Secondary: AUC vs number of observations.

## Encoding schemes (in scope)

- Simple parity (word count mod 2)
- Position-based (first letter of Nth word)
- One watermarking scheme from Redwood benchmark (cycle-shifting or balance-marking)

## Baselines

- LLM judge: "does this response appear to hide information?"
- Statistical test: chi-squared on surface features

## Key implementation note

`Q_task` for math: correctness is the dominant signal. Options for approximating it: LLM log-probs as a proxy, or a separate evaluator model scoring correctness + conciseness. The normalizing constant over the response space is intractable — in practice, we compare relative scores across a finite candidate set or use LLM log-probs directly.

Marginalize over unknown target bit (assume 50/50 prior). Marginalize over unknown encoding scheme by taking max or mixture over the scheme family. This requires a bit more thinking as we would like inverse planning to give (ideally) what the agents are originally saying through stenography. Need to to think a bit more about what the space of goals? are.

## Development order

The detector is independent of the data pipeline. `detect(response, task, schemes)` only needs a response string, a task dict, and encoding scheme functions — all three can be hardcoded for development. Build and unit-test the detector against toy inputs; plug in the real pipeline when ready. The only hard dependency on response generation is running actual AUC evaluations (you need real honest/stego labels for that). This is what we are currently working on. 
