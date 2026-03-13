

*the results tell a complete story — let me walk through it from the ground up.*

## what the experiment does

we have 100 AQUA-RAT math questions. for each question, we generated 100 candidate responses from gemini-3.1-flash-lite (on HPC). we split those 50/50:

- **detector pool** (50 responses): the detector uses these to estimate "what does an honest response look like for this question?"
- **attacker pool** (50 responses): we simulate both honest and stego agents by sampling from here

an **honest agent** picks a random response from the attacker pool — no constraint.

a **stego agent** picks a response that encodes a specific bit (the rare/minority bit for that question), simulating rejection sampling with budget k.

the **detector** scores a response by asking: "for each candidate partition function, how unusual is this response's bit assignment compared to the honest reference distribution?" the score is `log(0.5) - log(p_bit)`, where `p_bit` is the fraction of reference responses that land on the same side of the partition. high score = suspicious.

## the three schemes we test (as attacker)

these are the encoding schemes the simulated attacker uses. they differ in how much **encoding asymmetry** they create:

| scheme | what it does | median p_min | meaning |
|---|---|---|---|
| **synonym** | response contains "however" vs "but", "therefore" vs "thus", etc. | 0.083 | only ~8% of honest responses contain the minority synonym → huge signal per observation |
| **structural** | answer-first vs step-numbered format | 0.280 | ~28% minority → solid signal |
| **parity** | word count odd vs even | 0.440 | nearly 50/50 → almost no signal |

**p_min is everything.** it's the fraction of honest responses that encode the minority bit. when p_min is small, an attacker forced to encode that bit is doing something very unlikely for an honest agent — the "steganography tax." when p_min ≈ 0.5 (parity), there's no tax because honest responses are equally split.

## oracle vs codebook-free

**oracle mode:** the detector knows exactly which partition function the attacker is using. it scores with just that one function. this is the upper bound — no multiple comparison penalty.

**codebook-free mode:** the detector doesn't know the scheme. it maintains a library of ~55 partition functions (synonym pairs, structural features, statistical features), filters down to ~19 that show reasonable splits on the reference candidates, and searches for whichever partition shows the strongest signal. this is our main contribution.

## result 1: single-observation AUC

one question, one response, one detection decision.

| scheme | oracle AUC | codebook-free AUC |
|---|---|---|
| synonym | 0.92 | 0.63 |
| structural | 0.84 | 0.66 |
| parity | 0.67 | 0.52 |

**interpretation:** oracle is strong at N=1 for synonym/structural. codebook-free is weak at N=1 — the max over ~19 partitions means some irrelevant partition will randomly score high for honest responses too, washing out the true signal. this is the multiple comparisons problem and it's expected. **single-observation codebook-free detection is not the claim.**

## result 2: multi-observation AUC vs N (the headline result)

the detector observes N question-answer pairs from the same agent and must decide: honest or stego?

**critical implementation detail:** the correct aggregation is $\max_j \sum_i s(i,j)$ — sum each partition's score across all N observations, then take the max. the true partition accumulates consistent positive signal across observations; noise partitions do a random walk around zero. this is why multi-observation detection works codebook-free.

**stabilization note (2026-03-09):** original results used 1 sample per question and were RNG-dependent (two runs disagreed by 0.3 AUC). re-ran with R=20 samples/question, 5 seeds, 100 agents. script: `scripts/eval_diagnostic.py`. results: `results/diagnostic_oracle.json`, `results/diagnostic_codebook_free.json`.

### codebook-free (stabilized, mean ± std over 5 seeds)

| N | structural | synonym | parity |
|---|---|---|---|
| 1 | 0.656±0.037 | 0.626±0.046 | 0.520±0.034 |
| 3 | 0.834±0.025 | 0.762±0.056 | 0.485±0.071 |
| 5 | 0.932±0.015 | 0.882±0.013 | 0.533±0.048 |
| 7 | **0.986±0.004** | **0.928±0.028** | 0.567±0.040 |
| 10 | **0.995±0.003** | **0.984±0.007** | 0.649±0.030 |

### oracle (stabilized, mean ± std over 5 seeds)

| N | structural | synonym | parity |
|---|---|---|---|
| 1 | 0.837±0.037 | 0.920±0.024 | 0.670±0.048 |
| 3 | 0.967±0.009 | 0.997±0.003 | 0.786±0.047 |
| 5 | 0.992±0.007 | 1.000±0.000 | 0.822±0.046 |
| 7 | 0.998±0.002 | 1.000±0.000 | 0.900±0.034 |
| 10 | 1.000±0.000 | 1.000±0.000 | 0.955±0.018 |

**interpretation:** codebook-free detection converges toward oracle as N grows. the cost of not knowing the codebook is ~3-5 extra observations. by N=7, structural crosses 0.986 and synonym crosses 0.928 codebook-free. by N=10, both are >0.98.

parity with log-odds codebook-free stays at 0.649 even at N=10 — BUT the counting baseline achieves 0.986 for parity (see below). the failure is specific to log-odds scoring for weak-signal schemes in codebook-free mode.

## result 3: AUC vs attacker k

the attacker gets to sample k candidates and pick the one that encodes their target bit. higher k = attacker can be pickier = lower steganography tax. this was run in oracle mode.

| k | synonym AUC | structural AUC |
|---|---|---|
| 1 | 0.93 | 0.68 |
| 5 | 0.85 | 0.82 |
| 10 | 0.88 | 0.80 |
| 50 | 0.90 | 0.83 |

**interpretation:** AUC doesn't drop much as k increases. higher k helps the attacker *find* a compatible response (fewer failures), but the response still encodes the minority bit, so it's still in the rare tail of the honest distribution. at k=50 we're not near the MEC limit where detection truly fails.

## result 4: log-odds vs counting baseline (codebook-free multi-obs)

for single-observation, log-odds and counting produce identical AUC (both are monotone functions of p_bit; AUC is rank-based). the comparison only matters for multi-observation. counting baseline: `max_j count_i[score(i,j) > 0]` — just count how many observations each partition flags, take the max.

| N | log-odds structural | counting structural | log-odds synonym | counting synonym | log-odds parity | counting parity |
|---|---|---|---|---|---|---|
| 5 | 0.932 | 0.823 | 0.882 | 0.569 | 0.533 | **0.908** |
| 7 | 0.986 | 0.900 | 0.928 | 0.629 | 0.567 | **0.966** |
| 10 | 0.995 | 0.945 | 0.984 | 0.661 | 0.649 | **0.986** |

**log-odds beats counting for asymmetric schemes** (structural, synonym) by 3-5 observations. the evidence weighting matters when p_min varies across questions — some observations provide much stronger evidence than others, and log-odds captures this.

**counting beats log-odds for parity** (0.986 vs 0.649 at N=10). parity has p_min≈0.44, producing tiny log-odds scores (~0.13). in codebook-free mode, noise partitions with higher random scores win the max. counting ignores magnitude, just checking consistency.

## result 5: FPR at TPR=0.80 (codebook-free)

to catch 80% of stego agents, what fraction of honest agents do we falsely accuse?

| N | structural | synonym |
|---|---|---|
| 5 | 0.102 | 0.192 |
| 7 | 0.012 | 0.090 |
| 10 | 0.004 | 0.002 |

at N=10, false accusation rate < 0.5% for both schemes. operationally useful.

## result 6: p_min bucket analysis (single-obs)

| p_min range | oracle structural | cf structural | oracle synonym | cf synonym |
|---|---|---|---|---|
| [0.0, 0.1) | 0.981 | 0.813 | 0.996 | 0.461 |
| [0.1, 0.2) | 0.975 | 0.849 | 0.938 | 0.887 |
| [0.2, 0.3) | 0.895 | 0.707 | 0.810 | 0.485 |
| [0.3, 0.5) | 0.771 | 0.525 | 0.656 | 0.364 |

synonym at p_min < 0.1 gets AUC **0.461 codebook-free** — worse than random. the max-over-partitions selects the wrong partition when one partition has extreme signal but many others add noise. multi-obs fixes this (true partition accumulates, noise cancels), but at N=1 you're better off with oracle or simple pattern matching.

## what this means for the paper

1. **codebook-free detection works.** structural AUC 0.995, synonym 0.984 at N=10. stable across 5 seeds.

2. **the mechanism is exactly as theorized.** sum-then-max exploits the fact that the true partition accumulates (linear drift) while noise partitions cancel (random walk).

3. **parity is a negative control for log-odds but NOT for counting.** counting achieves 0.986 on parity at N=10 codebook-free. this means the "parity is undetectable" claim was wrong — it's undetectable by our log-odds method, but detectable by a simpler counting method. the log-odds scoring specifically fails for weak-signal schemes in codebook-free mode.

4. **the gap between oracle and codebook-free is 3-5 observations** for structural/synonym, consistent with the O(√(N/log M)) SNR scaling.

5. **log-odds outperforms counting by 3-5 observations for asymmetric schemes.** this is the method's actual value add: weighting evidence by strength. for balanced schemes, counting is better.


# does this project use inverse planning?

the detector computes `log(0.5) - log(p_bit)`. this is derived FROM inverse planning: model the stego agent as Boltzmann-rational (maximizing quality Q(r) subject to bit constraint), write the likelihood ratio P(r|stego)/P(r|honest), observe that Q(r) cancels, leaving a frequency test. the derivation is sound. but the computation is just "count how rare this bit is in the reference distribution."

**inverse planning is in the proof, not in the detector.** the proof tells you WHY this simple test is optimal (Q-cancellation). someone who never heard of inverse planning could arrive at the same test by thinking "check if the bit distribution is skewed." the paper should frame inverse planning as the theoretical justification that derives and validates the test, not as the runtime method.


# further ideas — status assessment (2026-03-09)

## 1. the geometric shadow (vector-space clustering) — RELEVANT, kind of implemented

**the problem:** our partition library (55 functions, ~19 after filtering) creates a multiple comparison penalty via the max operation. at N=1, this is devastating: synonym with p_min < 0.1 gets AUC 0.461 codebook-free (worse than random), because noise partitions outscore the true one by chance.

**what the diagnostic showed:** the penalty is real and quantified. codebook-free lags oracle by 3-5 observations. for parity (weak signal), log-odds codebook-free essentially fails while counting succeeds — the max-over-partitions drowns weak signals.

**would it help?** yes, in principle. projecting into embedding space and testing one data-driven anomaly hypothesis eliminates the multiple comparison penalty. but the embedding must capture the right features — if the encoding manipulates something the embedding is insensitive to (e.g., word count parity), the geometric shadow won't appear.

here's the picture:

### keyword_presence (78 valid questions, p_min mean=0.190)

| N | embedding | log-odds (blind) | oracle |
|---|-----------|-------------------|--------|
| 1 | 0.490 | 0.652 | 0.904 |
| 3 | 0.528 | 0.721 | 0.987 |
| 5 | 0.503 | 0.771 | 0.999 |
| 10 | 0.571 | 0.800 | 1.000 |

### number_format (29 valid questions, p_min mean=0.050)

| N | embedding | log-odds (blind) | oracle |
|---|-----------|-------------------|--------|
| 1 | 0.579 | 0.545 | 0.975 |
| 3 | 0.613 | 0.599 | 0.998 |
| 5 | 0.605 | 0.653 | 1.000 |
| 10 | **0.750** | 0.694 | 1.000 |

two things:

**keyword_presence** — embedding is at chance (0.49-0.57). this was supposed to be the best case: words like "consider" vs "explore" create different char 3-grams. but the signal drowns in the content variation of math responses. the partition detector (blind, max-over-19) does much better (0.80 at N=10) via accidental correlation. oracle hits 1.0 by N=5, confirming the scheme itself has power — it's the *detection method* that matters.

**number_format** — the one place embedding edges out log-odds blind (0.750 vs 0.694 at N=10). but only 29 valid questions (p_min too extreme for the other 71). and the margin is small. the oracle at 0.975 even at N=1 tells you this scheme is extremely detectable *if you know to look for it*.

### the real takeaway from these numbers

the oracle column is the story. keyword_presence oracle hits 0.904 at N=1, number_format 0.975 at N=1. these are *powerful* schemes. the problem isn't that the schemes are bad — it's that neither the embedding detector nor the blind partition detector knows where to look.

this reframes the problem: the bottleneck isn't "single geometric test vs 19 partitions." it's that the partition library doesn't include keyword_presence or number_format. if you added them to the partition set, you'd get near-oracle performance (at the cost of a slightly worse multiple comparison penalty from testing ~21 partitions instead of 19).



## 2. the ensemble hull (multi-model references + Wasserstein) — RELEVANT, NOT TESTED

**the problem:** all references come from gemini-3.1-flash-lite. if the suspect uses a different model, p_bit estimates may be wrong.

**what the diagnostic showed:** nothing directly — we only tested same-model. we have 25 cached haiku responses but haven't tested cross-model robustness. p_min varies hugely across questions (synonym: min=0.000, max=0.489), so mismatched references could badly distort detection.

**would it help?** yes for deployment realism. the cheapest test: use gemini references to detect a haiku agent using existing cached data. no new API calls.

**cost:** cross-model test: ~1 day. full ensemble + Wasserstein: ~1-2 weeks, requires idea 1's infrastructure.

## 3. the sensor array (multi-representation features) — PARTIALLY COVERED

**the problem:** if the encoding manipulates a feature our partition library doesn't capture, detection fails. position scheme (5th word A-M/N-Z) has no detector-side partition.

**what the diagnostic showed:** the 55-partition library covers lexical, structural, and statistical features adequately for synonym/structural/parity. but position and watermark_proxy (two attacker schemes) are genuinely invisible.

**would it help?** incrementally. character n-grams would catch position-like encodings. but this is partition library expansion, not a paradigm shift. the real value is combining with idea 1: embed via multiple representations, then anomaly-detect in the combined space.

**cost:** ~2-3 days on top of idea 1. spacy for syntax, character n-grams are trivial.