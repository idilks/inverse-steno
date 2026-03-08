# Literature Review: Viability of Inverse Planning for LLM Steganography Detection

## Research Question

Is the inverse planning detector (codebook-free, black-box, bit-frequency test with multi-observation compounding) a viable approach to detecting LLM steganography, given the landscape of known attacks and defenses?

---

## 1. Is Best-of-K / Distribution-Preserving Stego Standard Practice?

### What the literature says

There are three tiers of LLM steganography, ordered by sophistication:

**Tier 1: Naive / prompt-based encoding.** The attacker constrains the LLM via prompting (e.g., "start with phrase X if bit=1"). This is what Roger & Greenblatt (2023) evaluate, what Mathew et al. (2024) observe emerging from RL, and what our Redwood-style attackers implement. These schemes DO NOT preserve the output distribution. They impose a clear steganography tax. This is the tier where inverse planning works.

**Tier 2: Arithmetic-coding-based (AC, ADG, Discop).** The attacker has white-box access to the model's token-level probability distribution and uses arithmetic coding to embed bits by selecting tokens according to a modified distribution. These methods partially preserve the distribution but "inevitably distort" it (per the survey literature). KL divergence is nonzero but small. Detection is hard but not impossible — Bai et al. (2024) achieve 89-94% accuracy with fine-tuned LLM classifiers on AC/HC/ADG schemes, though this requires training on examples of the specific scheme family.

**Tier 3: Provably secure (MEC, Christ-Gunn-Zamir, SparSamp).** These schemes perfectly preserve the marginal output distribution. Schroeder de Witt et al. (2023) prove MEC-based stego is perfectly secure under Cachin's model. Christ, Gunn & Zamir (2024, TMLR) extend this to a computationally undetectable scheme. SparSamp (Wang et al., USENIX Security 2025) achieves this at 755 bits/sec with GPT-2. Against these, ANY black-box distributional test — including ours — is provably powerless. This is a theorem, not a conjecture.

### The practical gap

The key question is: do real-world colluding LLM agents actually USE Tier 3 methods?

**Current evidence says no.** Zolkowski et al. (2025) evaluated GPT-4.1, Claude-3.7, o3-mini and found that current models "nearly always fail" at message passing without CoT scratchpads. Even with coordination and pre-shared schemes, non-reasoning models succeed only ~20% of the time. MEC requires access to the model's own probability distribution — which API-based agents typically don't have (they get samples, not logits). SparSamp requires a seeded API (only some models expose this). The tokenization mismatch problem (subword segmentation ambiguity) causes decoding failures in practice and has only recently been partially addressed (SyncPool, arxiv 2508.20718).

**But this gap is closing.** The Adaptive Dynamic Sampling scheme (2026) achieves provably secure stego through API-only access using seed parameters. Several major LLMs (Qwen, GPT-4) already expose seed parameters. As models get more capable and APIs expose more control, Tier 3 becomes increasingly accessible.

### What this means for our project

**The inverse planning detector targets a real and currently-relevant threat class (Tier 1, partially Tier 2), but has a hard theoretical ceiling.** The experiment design already acknowledges this (Experiment 6 traces the boundary). The honest framing is: "we detect steganography that imposes a distributional tax, and we characterize exactly where detection breaks down." This is a valid contribution. The invalid framing would be claiming generality against all stego.

**Best-of-k rejection sampling** is the bridge between Tier 1 and Tier 3. In classical steganography, rejection sampling achieves provable security but with exponential time complexity in the number of bits per message. For 1 bit per response, the expected number of samples is 2 (trivial). For 4 bits, it's 16. For 10 bits, it's 1024. So a best-of-k attacker with modest k (say 8-16) can encode 3-4 bits with near-perfect distribution preservation. This is the most realistic near-term threat: not full MEC, but enough rejection sampling to substantially reduce the tax.

**Recommendation for Experiment 6:** Sweep k = {1, 2, 4, 8, 16, 32}. At k=1 (no rejection sampling), the tax is maximal. At k=32, a 1-bit scheme is effectively distribution-preserving. Plot AUC vs k. This traces the precise boundary of detectability.

---

## 2. Multiple Comparison Correction for Partition Testing

### The problem

We test M=50-200 candidate partitions and take the max score. Under the null (honest agent), the max of M correlated random variables is inflated. With Bonferroni, the threshold increases by log(M) ≈ log(200) ≈ 5.3 in log-odds. This is substantial — it means the per-observation signal must overcome a ~5 nat penalty before detection fires.

### What the literature says

**Bonferroni is the worst case.** It assumes independence between tests. Our candidate partitions are highly correlated — e.g., "word count mod 2" and "sentence count mod 2" will agree on many responses. Bonferroni massively overcorrects for correlated tests.

**Permutation testing with the max statistic is the gold standard.** Nichols & Holmes (2002, neuroimaging) established this: permute the data labels (honest vs. stego), recompute max_j S_j under permutation, build the null distribution of the max statistic empirically. This automatically accounts for the correlation structure between partitions. Empirical results show "near identical power to permutation tests and consistently more powerful than Bonferroni" even at M=200 correlated tests (Conneely & Boehnke 2007, PLOS Genetics).

**FDR (Benjamini-Hochberg) is inappropriate here.** FDR controls the fraction of false discoveries among all discoveries. We're making a single detection decision (stego or not), not discovering which partitions are active. FDR is for the scheme identification step, not the detection step.

**Multiscale testing** (Arias-Castro et al.) offers hierarchical partition search — discard large regions of hypothesis space cheaply, zoom in. This could help if the partition library has natural hierarchy (e.g., "lexical" → "synonym" → "specific pair").

### What this means for our project

**Use permutation testing, not Bonferroni.** Generate the null distribution of max_j S_j by permuting the stego/honest labels across the R bootstrap resamples. This is computationally cheap (we already compute S_j for all j) and strictly dominates Bonferroni.

**The correlation structure helps us.** If many partitions are correlated (as they will be — synonym-based partitions overlap, statistical partitions overlap), the effective number of independent tests is much less than M. A permutation test automatically captures this. The penalty might be equivalent to testing ~20-30 independent partitions rather than 200.

**Concrete implementation:**
1. For each permutation b = 1, ..., B: shuffle the honest/stego labels, compute D_b = max_j S_j
2. p-value = fraction of D_b >= D_observed
3. This gives exact control of the false positive rate at any threshold

**Power analysis sketch:** With M_eff ≈ 30 effective independent tests, the log-penalty is log(30) ≈ 3.4 nats. For a synonym scheme where p_bit ≈ 0.3 (30% of honest responses match the stego bit), the per-observation signal is log(0.5) - log(0.3) ≈ 0.51 nats. To overcome 3.4 + 1.65 (for 95% confidence) ≈ 5.1 nats, we need ~10 observations. This is feasible.

---

## 3. What Detection Approaches Work Against Sophisticated LLM Stego?

### The impossibility result

Against Tier 3 (distribution-preserving) stego, **no black-box output-level detector works.** This is proven:

- Schroeder de Witt et al. (2023): MEC stego is perfectly secure under Cachin's information-theoretic model
- Christ, Gunn & Zamir (2024): Computationally undetectable stego exists assuming one-way functions
- Westphal et al. (2026): Even perplexity and KL divergence fail — encoding 10 bits/token requires only ~0.026 nats expected KL, "much lower than expected during fine-tuning"

### What does work (partially)

**Gray-box / white-box approaches:**
- Westphal et al. (2026): Linear probes on model activations detect fine-tuning-based stego with 95-100% bit classification accuracy at final layers, vs. 60% for base models. But requires access to model internals.
- This is the only approach shown to work against distribution-preserving stego.

**Trained classifiers (Tier 2 only):**
- Bai et al. (2024): Fine-tuned Bloomz-7B/Llama-7B achieve 89-94% accuracy on AC/HC/ADG schemes. Does not require knowing the specific codebook, but IS trained on examples from the scheme family. Unclear generalization to unseen scheme families.
- Tang et al. (2024, LSGC): Only 67-68% accuracy on GPT-4-level LLM-generated stego. Substantially harder than classical stego.

**Paraphrasing defense (orthogonal to detection):**
- Roger & Greenblatt (2023): Context-aware paraphrasing limits encoding to ~3 bits/KB. But this is a defense (transform the text to destroy the channel), not detection (identify that the channel exists).
- Mathew et al. (2024): Paraphrasing reduces but doesn't eliminate 1-bit channels — >90% decode success persists for 1-bit payloads even after paraphrasing. "Standard steganography mitigations are not fully robust when optimized against."

**Monitoring / LLM judges (weak):**
- Zolkowski et al. (2025): GPT-4.1-mini monitor rated suspiciousness on 0-10 scale, calibrated for ≤1% FPR. Works against obvious encoding (acrostics, base64) but fails against subtle schemes.
- Motwani et al. (2024): "Limitations in countermeasures such as monitoring, paraphrasing, and parameter optimization."

### Where inverse planning fits

Our approach is unique in the literature: unsupervised (no stego training data), codebook-free, multi-observation compounding. No one else has done this. The closest comparisons:

| Approach | Requires training data? | Requires codebook? | Requires model weights? | Works against Tier 1? | Tier 2? | Tier 3? |
|---|---|---|---|---|---|---|
| **Inverse planning (ours)** | No | No | No | Yes | Partially | No |
| Bai et al. (2024) classifiers | Yes (scheme-family) | No | No | Yes | Yes | No |
| Westphal et al. (2026) probes | Yes | No | Yes (weights) | Yes | Yes | Partially |
| LLM judge monitor | No | No | No | Weak | No | No |
| Paraphrasing defense | No | No | No | N/A (defense) | N/A | N/A |

**The niche is real:** a black-box, unsupervised detector that compounds weak signals across observations. No published method occupies this cell. The limitation (fails against Tier 3) is shared by ALL black-box methods — it's a theorem, not our specific weakness.

---

## 4. The Redwood Text-Steganography-Benchmark

### What attacks it includes

Roger & Greenblatt (2023) evaluate these attack families:

1. **Watermark-based techniques:**
   - *Cycle-shifting*: Shifts token distribution cyclically based on message bits
   - *Balance-marking*: Marks tokens based on bit value

2. **Ad hoc prompt-based techniques:**
   - *Age instruction*: Changes first sentence based on message bit (instruction-level manipulation)
   - *Intro variation*: Selects among intro phrases to encode bits

3. **Model-recoverable techniques:**
   - *Acronym*: Hides message in first letters
   - *Scratchpad*: Uses a reasoning scratchpad to encode and later recover the message

**None of these use rejection sampling or distribution preservation.** They are all Tier 1 attacks. The watermark-based attacks (cycle-shifting, balance-marking) operate at the token-probability level but are NOT distribution-preserving — they deliberately shift the distribution to encode. The prompt-based attacks impose obvious distributional constraints.

### Our reimplementation

Our `src/benchmark/attackers.py` implements three Redwood-style attacks:
- **Intro variation**: 16 phrases, 4 bits encoded via phrase selection
- **Tone instruction**: 16 tone prompts, 4 bits encoded
- **Scratchpad** (implied from CLAUDE.md structure)

These are pure Tier 1 attacks. They WILL be detectable by our method because they impose measurable distributional constraints. The question is whether this is a meaningful benchmark or a strawman.

### Is this a strawman?

**Partially, but defensibly so.** The Redwood benchmark is the standard evaluation tool in the LLM stego safety literature. Zolkowski et al. (2025), Mathew et al. (2024), and Karpov et al. (2025) all use it or variants. Beating Tier 1 attacks with a novel unsupervised method is a valid first contribution. The paper becomes weak only if it claims or implies generality beyond Tier 1.

**What would strengthen it:** Include at least one Tier 2 attack (e.g., a simple arithmetic-coding-based scheme using logit access) and show degraded-but-nonzero detection. And include the Experiment 6 best-of-k sweep to precisely trace the detection boundary.

---

## Overall Assessment

### The case FOR viability

1. **The threat class is real and current.** Current LLMs can't do Tier 3 stego without special infrastructure (logit access, seeded APIs). Prompted/in-context encoding (Tier 1) is what actually emerges from RL training (Mathew et al. 2024) and what capability evals test (Zolkowski et al. 2025).

2. **The method is novel.** No one has applied inverse planning to stego detection. The Q-cancellation result (reduces to bit-frequency test) is clean. Multi-observation compounding is the right mechanism for weak-signal aggregation.

3. **The unsupervised property matters.** Bai et al.'s classifiers need training examples from the scheme family. Ours doesn't. In a real deployment, you don't know what scheme an agent might use — unsupervised detection is the correct threat model.

4. **Multiple comparison correction is tractable.** Permutation testing with correlated partitions reduces the effective penalty to ~log(30) rather than log(200). Power analysis suggests N=10-20 observations suffices for typical Tier 1 schemes.

### The case AGAINST viability

1. **The theoretical ceiling is low.** Any distribution-preserving scheme (MEC, Christ-Zamir, SparSamp) defeats all black-box detectors including ours. This isn't a limitation of our method — it's a theorem. But it means the method has a well-defined expiration date as stego techniques improve.

2. **Best-of-k closes the gap fast.** For 1-bit encoding, k=2 samples cuts the distributional deviation roughly in half. k=8 makes it nearly invisible. Rejection sampling is trivial to implement and doesn't require logit access. A minimally sophisticated attacker can dramatically reduce the tax for low bandwidths.

3. **The Redwood benchmark is not hard enough.** All Redwood attacks are Tier 1. Beating them with a statistical test is expected. The interesting question — does inverse planning detect anything that simpler methods don't? — requires harder attacks.

4. **Reference model mismatch is underexplored.** The detector generates reference responses from "the unconstrained model." In practice, the detector may not have access to the exact same model the agent uses. Model mismatch creates systematic bias in p_bit estimates, potentially generating false positives or masking true positives. This is an engineering problem but a serious one.

5. **Adaptive attackers.** If the attacker knows the partition library, they can design schemes that are orthogonal to all tested partitions. The "learned partitions" (embedding clustering) provide some defense but may be dominated by irrelevant variation (topic, length) rather than capturing the stego boundary.

### Where the evidence actually points

**The method is viable for the current threat landscape but has a clear expiration date.** The contribution is:

- A novel detection framework (inverse planning → Q-cancellation → bit-frequency test)
- First unsupervised, codebook-free stego detector with multi-observation compounding
- A precise characterization of the detection boundary (AUC vs attacker-k curve)

The contribution is NOT:
- A general-purpose stego detector
- A solution to the agent collusion problem against sophisticated attackers
- Effective against distribution-preserving stego (by theorem)

**Confidence: moderate-high** that the method works on Tier 1 attacks. **Confidence: low** that it provides meaningful detection against Tier 2+ attacks with even modest distribution preservation.

### Key uncertainties

1. **How quickly does best-of-k kill detection power?** This is the single most important experiment. If AUC drops below 0.7 at k=4, the method's practical scope is very narrow.

2. **Does the permutation-based multiple comparison correction preserve enough power at M=200?** Needs empirical measurement on the actual partition library.

3. **How sensitive is detection to reference model mismatch?** If the agent uses a different model (or even different temperature) than the reference generator, what happens to false positive rates?

4. **Do learned partitions (embedding clustering) detect anything the hand-crafted partitions miss?** If not, the "codebook-free" claim weakens to "codebook-library-dependent."

5. **Will the gap between Tier 1 and Tier 3 persist?** If API-based provably-secure stego becomes easy (e.g., seed-based ADS schemes become standard), the entire Tier 1 threat class may become irrelevant within 1-2 years.

---

## Bibliography

### Provably Secure / Distribution-Preserving Steganography
- Schroeder de Witt, C., et al. "Perfectly Secure Steganography Using Minimum Entropy Coupling." ICLR 2023.
- Christ, M., Gunn, S., Zamir, O. "Undetectable Watermarks for Language Models." COLT 2024 / TMLR 2024.
- Zamir, O. "Undetectable Steganography for Language Models." TMLR, Oct 2024.
- Wang, Y., et al. "SparSamp: Efficient Provably Secure Steganography Based on Sparse Sampling." USENIX Security 2025.
- OD-Stega. "LLM-Based Near-Imperceptible Steganography via Optimized Distributions." ICLR 2025.
- Adaptive Dynamic Sampling. "Provable Secure Steganography Based on Adaptive Dynamic Sampling." arXiv 2504.12579, 2026.
- SyncPool. "Addressing Tokenization Inconsistency in Steganography and Watermarking Based on LLMs." arXiv 2508.20718, 2025.

### Detection and Steganalysis
- Bai, Y., et al. "Towards Next-Generation Steganalysis: LLMs Unleash the Power." 2024.
- Tang, L., Liu, Z., et al. "Rethinking LLM and Linguistic Steganalysis (LSGC)." 2024.
- Westphal, C., Navaie, K., Rosas, F. "Hide and Seek in Embedding Space: Geometry-based Steganography and Detection in LLMs." arXiv 2601.22818, Jan 2026.

### Agent Collusion and AI Safety
- Motwani, S., Schroeder de Witt, C., et al. "Secret Collusion among AI Agents: Multi-Agent Deception via Steganography." NeurIPS 2024.
- Mathew, A., Matthews, M., et al. "Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs." AACL-IJCNLP 2025.
- Roger, F., Greenblatt, R. "Preventing Language Models From Hiding Their Reasoning." arXiv 2310.18512, 2023.

### Capability Evaluations
- Zolkowski, A., et al. "Early Signs of Steganographic Capabilities in Frontier LLMs." arXiv 2507.02737, 2025.
- Karpov, N., et al. "Steganographic Potentials of Language Models." ICLR Building Trust Workshop, 2025.
- "LLMs can learn and generalize steganographic CoT under process supervision." NeurIPS 2025.

### Multiple Comparison Correction
- Nichols, T., Holmes, A. "Nonparametric Permutation Tests for Functional Neuroimaging." Human Brain Mapping, 2002.
- Conneely, K., Boehnke, M. "Rapid and Accurate Multiple Testing Correction and Power Estimation for Millions of Correlated Markers." PLOS Genetics, 2007.

### Steganographic Theory
- Cachin, C. "An Information-Theoretic Model for Steganography." Information and Computation, 1998.
- Hopper, N., Langford, J., von Ahn, L. "Provably Secure Steganography." CRYPTO 2002.
- Lysyanskaya, A., Meyerovich, M. "Provably Secure Steganography with Imperfect Sampling." 2006.

### LLM Steganography Methods
- Wu, J., et al. "Generative Text Steganography with Large Language Model (LLM-Stega)." ACM MM 2024.
