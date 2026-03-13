
Detecting covertext steganography in LLM outputs via inverse planning. A colluding agent encoding a hidden bit restricts its outputs to bit-compatible ones, paying a steganography tax that's statistically detectable.

Benchmark: [Redwood Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark). Dataset: AQUA-RAT.

## Structure

```
src/                          # detector side (Idil)
  schemes/                    # pure decode functions: str -> int | None
    synonym.py                # first synonym-pair match -> bit (16 pairs, PRIMARY)
    structural.py             # answer position + step numbering -> bit
    parity.py                 # word count mod 2 (sanity check only)
    recommendation.py         # buy/sell signal -> bit
    keyword_presence.py       # cognitive verbs, transitions, etc. -> bit (5 regex categories)
    number_format.py          # numeric formatting choices (commas, %, fractions) -> bit (8 signals)
    partitions.py             # 50+ @partition-decorated functions for codebook-free detection + filter_partitions()
  detector/
    inverse_planning.py       # detect(), detect_multi(), decode_message()
    sampling.py               # async candidate generation + disk cache
    embedding.py              # TF-IDF char 3-gram + SVD anomaly detection (kNN/Mahalanobis/cosine), LOO p-values, Fisher's method
    embedding_cache.py        # disk cache for embeddings (.npz), keyed by (text_hash, method)
  baselines/
    llm_judge.py              # prompt-based detection
    chi_squared.py            # frequency test on agent's bit distribution
  eval/
    auc.py                    # Mann-Whitney AUC
    multi_obs.py              # AUC-vs-N curve
    permutation.py            # permutation testing
    run_eval.py               # evaluation from JSON data files + benchmark results
  benchmark/
    attackers.py              # 3 Redwood-style attackers (intro, scratchpad, tone), 4 bits each
    dataset.py                # AQUA-RAT loader (HuggingFace + local cache)
    run_benchmark.py          # full pipeline: both conditions, AUC + BER metrics

scripts/                      # evaluation & experiment scripts
  eval_diagnostic.py          # stabilized diagnostic eval (oracle + codebook-free, multi-obs)
  eval_aqua.py                # AQUA-RAT evaluation
  eval_blind_schemes.py       # embedding detector on schemes with no partition (watermark_proxy, position)
  eval_new_schemes.py         # embedding + partition on keyword_presence, number_format
  bridge_eval.py              # end-to-end: attacker data -> reference gen -> partition filter -> detection + AUC
  experiment_auc_vs_k.py      # AUC as function of attacker pool size K
  experiment_multi_obs.py     # multi-observation AUC-vs-N curves
  generate_aqua_candidates.py # bulk candidate generation for AQUA-RAT questions
  validate_partitions.py      # partition health check: GOOD/WEAK/DEAD/ONE_SIDED classification
  show_distributions.py       # visualization: reference bit distributions, honest vs stego accumulation
  plot_paper_figures.py       # AAAI paper figures (AUC-vs-N, score accumulation, example interaction)

agents/                       # attacker/data-gen side
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
results/                      # experiment results (JSON)

tests/                        # tests!
```

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v
python main.py --model stub --n-tasks 10    # attacker-side data generation
```

## Scheme agreement (detector <-> attacker)

Both sides use same bit mapping for synonym (first element = bit 0), structural (Step1 = bit 0), parity (word_count % 2). Two attacker schemes (position, watermark_proxy) have no detector-side equivalent. 

## Existing data

`output/generated/dataset.jsonl`: 60 records, Haiku-4.5, synthetic arithmetic. 100% round-trip success, 100% task correctness. Tasks are too easy to show steganography tax on accuracy — real AQUA-RAT needed for meaningful experiment.

## Stabilized results 

100 AQUA-RAT questions, gemini-3.1-flash-lite, 100 candidates/question (50 detector, 50 attacker pool), R=20 samples/question, 5 seeds, 100 simulated agents. Results in `results/diagnostic_oracle.json` and `results/diagnostic_codebook_free.json`.

**Codebook-free multi-obs AUC (mean ± std):**

| N | structural | synonym | parity |
|---|---|---|---|
| 5 | 0.932±0.015 | 0.882±0.013 | 0.533±0.048 |
| 7 | 0.986±0.004 | 0.928±0.028 | 0.567±0.040 |
| 10 | 0.995±0.003 | 0.984±0.007 | 0.649±0.030 |

**Key findings:**
- Codebook-free detection works and is stable across seeds: structural 0.995, synonym 0.984 at N=10. 
- Counting baseline beats log-odds for parity (0.986 vs 0.649 at N=10 codebook-free). Log-odds wins for asymmetric schemes.
- Log-odds advantage over counting is ~3-5 fewer observations to reach same AUC for structural/synonym per example.
- FPR@TPR=0.80: structural 0.004, synonym 0.002 at N=10
