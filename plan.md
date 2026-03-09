*the detector works — the experiment harness doesn't yet ask the right question.*

## Fix Plan

### Fix 1: Codebook-free detection (the multiple comparisons problem)

**root cause:** `detect()` returns `max(scores)` across all schemes. with M=55 partitions, some noisy partition always fires high for both honest and stego responses. the signal (structural AUC=1.0) drowns in the noise floor of 54 irrelevant partitions.

**fix:** change codebook-free mode to use per-partition p-values instead of raw max score.

- file: `scripts/bridge_eval.py`, in `run_bridge_eval()`
- when `mode == "codebook-free"`:
  1. for each record, call `partition_pvalues()` from `src/eval/permutation.py` (already written) — this computes a permutation p-value for each partition
  2. apply `benjamini_hochberg()` (already written) to get corrected p-values
  3. detection score = `-log(min adjusted p-value)` — high when any partition is significant after correction
  4. this replaces the current `detect()` call for codebook-free mode
- **cost:** ~1000 permutations × 55 partitions × 60 records. no API calls, but computationally heavier. ~30s-1min. could reduce `n_perms` to 200 for speed.
- alternatively: a simpler approach — prefilter partitions using the reference candidates (drop ONE_SIDED and DEAD), reducing M from 55 to ~19. then run `detect()` with the filtered set. this alone might recover most of the signal without permutation testing. try this first.

**implementation steps:**
1. add `filter_partitions(candidates, schemes, min_split=0.1, max_none=0.5)` utility to `partitions.py` — returns only partitions with reasonable splits on the reference candidates
2. in `bridge_eval.py` codebook-free mode, filter partitions per-question using the reference candidates before calling `detect()`
3. if filtered-max still fails, switch to the BH-corrected p-value approach

### Fix 2: AUC-vs-k experiment (shared pool contamination)

**root cause:** the attacker samples its response from the same 50 candidates that the detector uses as references. so the "stego" response is literally a member of the reference set — it scores exactly like an honest response. the detector is comparing a candidate against itself.

**fix:** split candidates into two non-overlapping pools.

- file: `scripts/experiment_auc_vs_k.py`, in `run_experiment()`
- generate K=100 candidates total (or use K=50 and split)
- pool A (25 candidates): detector's reference set
- pool B (25 candidates): attacker's sampling pool
- attacker draws k from pool B, picks one encoding target bit
- detector scores against pool A only
- this correctly simulates the setting where attacker and detector use independent samples from the same model

**implementation steps:**
1. change `--k-ref` default to 100, or add `--k-attacker` parameter
2. in `run_experiment()`, split `candidates` into `detector_refs = candidates[:k//2]` and `attacker_pool = candidates[k//2:]`
3. pass `detector_refs` to `detect()`, `attacker_pool` to `simulate_attacker_response()`
4. the 50 cached candidates can be the detector refs; need 50 more for attacker pool (500 more API calls, or just use k=50 split 25/25 for now)

### Fix 3: increase statistical power (optional, for after fixes 1-2)

the synonym scheme has p_bit ≈ 0 for bit=1 on most questions — Haiku never uses "thus", "compute", etc. this means:
- stego responses targeting bit=1 are *maximally detectable* (log_odds = log(0.5) - log(1/52) ≈ 3.26)
- but stego responses targeting bit=0 are *invisible* (the common bit)
- and the attacker in `experiment_auc_vs_k` randomly picks target_bit, so half the stego responses are undetectable by design

fix: in the experiment, only simulate attacker targeting the *rare* bit (this is the adversarial setting — the attacker needs to encode information, not just match the majority). or: simulate encoding a message (alternating 0s and 1s) and use multi-observation detection.

---

## Next Steps (after clearing context)

1. **implement fix 1** (codebook-free partition filtering) — should take the codebook-free AUC from 0.49 to something meaningful
2. **implement fix 2** (split pools in AUC-vs-k) — needs either 500 more API calls or a 25/25 split of existing cache
3. **re-run all three experiments** with fixes applied
4. **run on real AQUA-RAT** — the synthetic arithmetic questions are trivially easy, producing very uniform response formats. real AQUA-RAT will have more variation in structure and word choice, which means more balanced p_bit splits and harder detection. this is where the HPC comes in (100+ questions × 50-100 candidates = 5000-10000 API calls)
5. **multi-observation experiment** — use `detect_multi()` to test whether aggregating across N questions pushes AUC above 0.9 for parity/synonym (the oracle result estimates N~107 for parity)

**key results so far to carry forward:**
- oracle structural AUC = 1.0 (perfect, p_bit ≈ 0 for bit=1)
- oracle synonym AUC = 0.7 (good, but one-sided — only bit=1 is detectable)  
- oracle parity AUC = 0.6 (weak single-obs, needs N~100 multi-obs)
- reference candidates cached in `cache/samples/` for all 10 questions
- results saved in `results/bridge_oracle.json` and `results/bridge_codebook-free.json`