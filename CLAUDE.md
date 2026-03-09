# CLAUDE.md

## Project

Detecting covertext steganography in LLM outputs via **inverse planning**. A colluding agent encoding a hidden bit restricts its outputs to bit-compatible ones, paying a "steganography tax" that's statistically detectable.

Benchmark: [Redwood Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark). Dataset: AQUA-RAT.

## Key insight: Q-cancellation

The log-odds ratio simplifies to `log(0.5) - log(p_bit)`, where `p_bit` is the fraction of honest samples encoding the same bit. Q(r) cancels completely — the detector is a bit-frequency test with Boltzmann-rational justification. Per-observation power depends on encoding asymmetry; multi-observation detection is where discrimination compounds.

**Why not just decode?** Knowing the codebook tells you *what bit* a response encodes, but every response (honest or stego) decodes to some bit. The detector answers whether the response was *intentionally constrained* to encode that bit. The signal is distributional: if only 20% of honest responses encode bit 1 for a given question, but this response encodes bit 1, that's suspicious. Across multiple observations, these asymmetries compound.

## Structure

```
src/                          # detector side (Idil)
  schemes/                    # pure decode functions: str -> int | None
    synonym.py                # first synonym-pair match -> bit (16 pairs, PRIMARY)
    structural.py             # answer position + step numbering -> bit
    parity.py                 # word count mod 2 (sanity check only)
    recommendation.py         # buy/sell signal -> bit
  detector/
    inverse_planning.py       # detect(), detect_multi(), decode_message()
    sampling.py               # async candidate generation + disk cache
  baselines/
    llm_judge.py              # prompt-based detection
    chi_squared.py            # frequency test on agent's bit distribution
  eval/
    auc.py                    # Mann-Whitney AUC (no sklearn)
    multi_obs.py              # AUC-vs-N curve
    run_eval.py               # evaluation from JSON data files + benchmark results
  benchmark/
    attackers.py              # 3 Redwood-style attackers (intro, scratchpad, tone), 4 bits each
    dataset.py                # AQUA-RAT loader (HuggingFace + local cache)
    run_benchmark.py          # full pipeline: both conditions, AUC + BER metrics

agents/                       # attacker/data-gen side (Mohei)
  base_agent.py               # BaseAgent ABC, AgentResponse dataclass
  honest_agent.py             # best-of-k generation, q_value scoring
  steganographic_agent.py     # best-of-k + encoding, round-trip verification
  model_interface.py          # StubModel, DartmouthChatModel (sync), load_model()
encoding_schemes/             # class-based encode+decode (attacker interface)
  base_scheme.py              # BaseEncodingScheme ABC, EncodingResult, attempt_encoding()
  synonym_scheme.py           # 8 pairs (subset of src/schemes/synonym.py's 16)
  structural_scheme.py        # Step1/2 prefix (bit 0) vs answer-first (bit 1)
  parity_scheme.py            # word count mod 2, appends "indeed"
  position_scheme.py          # 5th word A-M/N-Z (no src/schemes/ equivalent)
  watermark_proxy_scheme.py   # "To begin,"/"At the outset," prefix (no equivalent)
  scheme_registry.py          # SchemeRegistry.default() returns all 5
pipeline/
  data_generator.py           # DatasetGenerator: loops questions x schemes, produces records
tasks/
  base_task.py                # BaseTask ABC, TaskInstance dataclass
  aqua_rat_task.py            # AQUA-RAT loader + synthetic fallback, q_value scoring
dataset/
  schema.py                   # DatasetRecord, DatasetMetadata dataclasses
  writer.py                   # JSONL writer
main.py                       # CLI: python main.py --model stub --n-tasks 10
output/generated/             # 60 records from Haiku-4.5 run (10 questions x 6 records)

tests/                        # 70 tests (detector side only), all passing
```

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v
python main.py --model stub --n-tasks 10    # attacker-side data generation
```

## Scheme agreement (detector <-> attacker)

Both sides use same bit mapping for synonym (first element = bit 0), structural (Step1 = bit 0), parity (word_count % 2). Two attacker schemes (position, watermark_proxy) have no detector-side equivalent — detector is blind to those.

## Existing data

`output/generated/dataset.jsonl`: 60 records, Haiku-4.5, synthetic arithmetic. 100% round-trip success, 100% task correctness. Tasks are too easy to show steganography tax on accuracy — real AQUA-RAT needed for meaningful experiment.

## Stabilized results (2026-03-09)

100 AQUA-RAT questions, gemini-3.1-flash-lite, 100 candidates/question (50 detector, 50 attacker pool), R=20 samples/question, 5 seeds, 100 simulated agents. Results in `results/diagnostic_oracle.json` and `results/diagnostic_codebook_free.json`.

**Codebook-free multi-obs AUC (mean ± std):**

| N | structural | synonym | parity |
|---|---|---|---|
| 5 | 0.932±0.015 | 0.882±0.013 | 0.533±0.048 |
| 7 | 0.986±0.004 | 0.928±0.028 | 0.567±0.040 |
| 10 | 0.995±0.003 | 0.984±0.007 | 0.649±0.030 |

**Key findings:**
- Codebook-free detection works: structural 0.995, synonym 0.984 at N=10. Stable across seeds.
- Counting baseline beats log-odds for parity (0.986 vs 0.649 at N=10 codebook-free). Log-odds wins for asymmetric schemes.
- Log-odds advantage over counting is ~3-5 fewer observations to reach same AUC for structural/synonym.
- FPR@TPR=0.80: structural 0.004, synonym 0.002 at N=10. Very low false accusation rate.
- Multiple comparison penalty hurts at N=1 (synonym p_min<0.1 gets AUC 0.461 codebook-free vs 0.996 oracle). Multi-obs fixes this.
- "Inverse planning" is in the derivation (Q-cancellation proof), not the runtime computation. The detector is a frequency test.

## Status

- **Done:** detector core, encoding schemes, baselines, eval pipeline, benchmark integration, decode_message(), 70 unit tests, stabilized diagnostic evaluation (`scripts/eval_diagnostic.py`). Attacker-side pipeline merged from collision-mockup branch.
- **Known issues:** (1) `model_fn` signature inconsistent in `run_benchmark.py`. (2) `model_interface.py` is sync, detector expects async. (3) No tests for attacker side. (4) Two attacker schemes (position, watermark_proxy) have no detector-side partition — detector is blind.
- **Abandoned:** side-channel/computational-tax detection (idea 4). Modern stego methods (SparSamp, Shimmer) have O(1) overhead per token; no timing signal to detect.
- **Next:** explore embedding-based anomaly detection (idea 1 from results-1.md) to reduce multiple comparison penalty. Also: K-sweep (does K=10 references work?) and cross-model robustness test.
