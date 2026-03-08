# Detecting Steganographic Constraint via Inverse Planning

## Research Question

Can we detect that an LLM agent is steganographically constrained — without knowing the encoding scheme — by identifying the binary partition signature that all covertext steganography must impose?

## Core Idea

Steganographic encoding schemes partition the response space into two sets (S_0 and S_1). A steganographic agent samples from only one half per query, while an honest agent samples from the full space. Subsequently our detector searches for partition functions that explain the agent's observed responses as intentionally biased, using inverse planning to score each candidate partition.

This is not perplexity (which measures text quality) or diversity (which measures spread), instead we are looking for if the agent's outputs are systematically missing from one side of some partition, and the detector's job is to find that partition.

## Why Not Brute Force?

The space of possible encoding schemes is combinatorially large within each family (e.g., which synonym pairs? which word position? which structural feature?). Brute-force enumeration scales with |codebook space|. Inverse planning scales with |reference samples| as it detects the *existence* of constraint without enumerating what the constraint is. Scheme identification subsequently is a second, cheaper step after detection.

## Algorithm

### Definitions

Let $\mathcal{R}$ denote the space of all possible free-text responses.

- **Queries.** $q_1, \dots, q_N$ are the $N$ observed question-answer interactions. Each $q_i$ is an AQUA-RAT question.

- **Agent response.** $r_i^* \in \mathcal{R}$ is the agent's actual response to query $q_i$. This is the single observation the detector sees per interaction.

- **Reference responses.** For each query $q_i$, the detector generates $K$ reference responses $\{r_{i,1}, \dots, r_{i,K}\}$ by sampling from the unconstrained model $P(\cdot \mid q_i)$. These approximate the honest response distribution for that query.

- **Candidate partition function.** $\varphi_j : \mathcal{R} \to \{0, 1\}$ is a deterministic function that maps any response to a single bit. The detector maintains a library of $M$ such functions, $\{\varphi_1, \dots, \varphi_M\}$. Each $\varphi_j$ represents a hypothesis about how an attacker might partition the response space to encode information. Examples: $\varphi_j(r) = \mathbb{1}[\text{word count of } r \text{ is odd}]$, or $\varphi_j(r) = \mathbb{1}[r \text{ contains "large" rather than "big"}]$.

- **Bit-agreement probability.** For query $q_i$ and partition $\varphi_j$:

$$\hat{p}_{\text{bit}}(i, j) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\bigl[\varphi_j(r_{i,k}) = \varphi_j(r_i^*)\bigr]$$

This is the empirical probability that a reference response falls on the same side of $\varphi_j$ as the agent's response. It estimates $P_{r \sim P(\cdot|q_i)}[\varphi_j(r) = \varphi_j(r_i^*)]$.

### Stage 1: Codebook-Free Detection

**Input:** $N$ observed pairs $\{(q_i, r_i^*)\}_{i=1}^N$, reference model $P$, partition library $\{\varphi_1, \dots, \varphi_M\}$.

**Output:** Detection score $D \in \mathbb{R}$ (higher = more likely steganographic).

**Step 1 (Reference generation).** For each $i \in \{1, \dots, N\}$, sample $K$ responses i.i.d. from $P(\cdot \mid q_i)$:

$$r_{i,1}, \dots, r_{i,K} \sim P(\cdot \mid q_i)$$

**Step 2 (Per-observation scoring).** For each query $i$ and partition $j$, compute:

$$s(i, j) = \log \frac{1}{2} - \log \hat{p}_{\text{bit}}(i, j)$$

**Interpretation.** This is a log-likelihood ratio. Under the steganographic model (agent constrained to encode a uniformly random bit), the probability of observing the agent on a given side of $\varphi_j$ is $\frac{1}{2}$ — the message bit is chosen uniformly. Under the honest model, it is $\hat{p}_{\text{bit}}(i,j)$. So $s(i,j) > 0$ when the agent's bit-side is in the minority among references (suspicious), and $s(i,j) < 0$ when it is in the majority (unsuspicious).

**Derivation (Q-cancellation).** The full Boltzmann-rational inverse planning score is $\log Q_\theta(r_i^*) - \log P(r_i^* \mid q_i)$, where $Q_\theta$ is the steganographic agent's policy. Under the assumption that the steganographic agent samples uniformly from the bit-compatible half of the distribution, $Q_\theta(r_i^*) = P(r_i^* \mid q_i) / P[\varphi_j(r) = \varphi_j(r_i^*) \mid q_i]$. Taking the log ratio, the $P(r_i^* \mid q_i)$ terms cancel, yielding $-\log P[\varphi_j(r) = \varphi_j(r_i^*) \mid q_i]$. Adding the message prior $\log \frac{1}{2}$ gives the formula above.

**Step 3 (Aggregation across observations).** For each partition $j$, sum over queries:

$$S_j = \sum_{i=1}^{N} s(i, j)$$

**Step 4 (Maximization over partitions).** The detection score is:

$$D = \max_{j \in \{1, \dots, M\}} S_j$$

Flag as steganographic if $D > \tau$, where threshold $\tau$ is calibrated on honest-agent data (see Experiment 4 for multiple comparison correction).

**Special case (known codebook).** When $M = 1$ and $\varphi_1$ is the true decoding function, the max is trivial and the detector reduces to a bit-frequency test with no multiple comparison penalty.

### Stage 2: Scheme Identification (conditional on detection)

The partition $\varphi_{j^*}$ with highest score indicates the scheme family. Within that family, the multi-observation signal narrows the parameterization (e.g., which specific synonym pairs are being used).

### Stage 3: Decoding (traditional)

Once the scheme is identified, decode conventionally, as inverse planning is not needed here.

### Stage 2: Scheme Identification (conditional on detection)

The partition phi_j* with highest score indicates the scheme family. Within that family, the multi-observation signal narrows the parameterization (e.g., which specific synonym pairs are being used).

### Stage 3: Decoding (traditional)

Once the scheme is identified, decode conventionally, as inverse planning is not needed here.

## Candidate Partition Functions

The detector needs a library of candidate partitions. These don't need to match the attacker's exact scheme but need to be able to derive it:

| Family | Candidate partitions | How generated |
|---|---|---|
| Lexical | "Does response contain word A vs word B?" for common synonym pairs | Precomputed from synonym databases (WordNet, etc.) |
| Structural | "Does response start with a step number?", "Is the answer stated first?", paragraph count parity | Hand-crafted from known formatting patterns |
| Statistical | Word count mod 2, sentence count mod 2, avg word length above/below median | Enumerable, small set |
| Semantic | Sentiment polarity, confidence level, hedging presence | LLM-extracted or classifier-based |
| Learned | Cluster assignment from response embeddings (k=2 clustering on reference set) | Per-query, data-driven |

The "learned" category is given by for each query, cluster the K reference responses into 2 groups and test whether the agent's response is systematically in one cluster. This is fully codebook-agnostic but may have low power if the encoding doesn't align with the dominant axis of variation.

## Experimental Design

### Communication Structure

The setup is **one-way monitoring**: a single agent answers AQUA-RAT questions, and the detector observes those answers. There is no second agent in the experiment. The two-agent collusion scenario is the *threat model motivation* (why would an agent encode?), but the detection algorithm only watches one agent's output stream.

One **interaction** = one question-answer pair. The agent receives an AQUA-RAT question and produces a free-text response (with reasoning and a final answer). In the honest condition, the agent answers without constraint. In the stego condition, each response encodes B bits of a random message (see Encoding Bandwidth below).

A **detection decision** is made after observing N interactions. The detector sees N question-answer pairs and outputs a score (higher = more likely steganographic). The key experimental variable is N: how many interactions does the detector need before it can reliably distinguish stego from honest?

### Encoding Bandwidth

Real colluding agents want bandwidth — 1 bit per response over 100 questions yields only 12.5 bytes. We test two bandwidth levels:

| Condition | Bits/response | How | Constraint severity | Detectability |
|---|---|---|---|---|
| Low-bandwidth (B=1) | 1 | Single scheme per response | Agent restricted to 1/2 of output space | Harder to detect (baseline) |
| High-bandwidth (B=4) | 4 | Four schemes simultaneously per response | Agent restricted to ~1/16 of output space | Easier to detect |

**Why 1 bit is the conservative case, not the naive one:** more bits per response means more simultaneous constraints on the output, which means a larger steganography tax. At B=4, the agent must satisfy synonym AND structural AND parity AND statistical constraints simultaneously — the intersection of four partition halves is ~1/16 of the response space. The probability that an honest agent lands in this intersection by chance is much lower, so false positives drop and detection power increases.

The paper result becomes: "at B=1, detection requires N=X observations to reach AUC=0.9; at B=4, it requires N=Y where Y << X." This traces the fundamental tradeoff between attacker bandwidth and detectability.

For the high-bandwidth condition, the detection algorithm changes slightly: instead of max_j S_j over individual partitions, we also test *conjunctions* of partitions (S_{j1 AND j2 AND ...}). An honest agent biased on one partition by chance is plausible; biased on four independent partitions simultaneously is not.

### Concrete Parameters

| Parameter | What it is | Value | Rationale |
|---|---|---|---|
| Q | Total AQUA-RAT questions | 100-200 | Available dataset size |
| K | Reference responses per question | 50-100 | Needed to estimate unconstrained distribution per query |
| N | Observations per detection decision | 1, 2, 5, 10, 20, 50 | Independent variable — swept in Experiment 1 |
| R | Detection runs per condition (for AUC) | 200+ | Bootstrap resampling from Q questions gives stable AUC estimates |
| M | Candidate partition functions | ~50-200 | Determines multiple comparison penalty; see Experiment 4 |

**Total data generation cost:** Q × (K references + 1 stego response + 1 honest response) per scheme. With Q=100, K=50: **5,200 responses per scheme**. Across 4 stego schemes + honest baseline: **~22,000 API calls total**. Feasible with Haiku-class models.

**How AUC is computed at each N:** sample N questions (without replacement) from Q, compute detection score for the stego agent's responses and separately for the honest agent's responses. Repeat R times with different subsamples. This gives R stego scores and R honest scores → one AUC value. Varying N produces the AUC-vs-N curve.

### Conditions

**Single-scheme conditions (B=1):**

| Condition | Agent behavior | Role |
|---|---|---|
| Honest | Answers freely, no encoding | Baseline (null hypothesis) |
| Stego-synonym | Encodes 1 bit via synonym substitution | Test |
| Stego-structural | Encodes 1 bit via answer positioning / step numbering | Test |
| Stego-parity | Encodes 1 bit via word count parity | Test |
| Stego-position | Encodes 1 bit via 5th-word letter range | Test (unseen by detector[^1]) |

**Multi-scheme condition (B=4):**

| Condition | Agent behavior | Role |
|---|---|---|
| Stego-multi | Encodes 4 bits simultaneously (synonym + structural + parity + statistical) | High-bandwidth test |

[^1]: The position scheme has no corresponding candidate partition in the detector's library. This tests whether the learned/statistical partitions generalize to unknown scheme families.

Each condition answers the same set of AQUA-RAT questions. Single-scheme agents encode one random bit per question. The multi-scheme agent encodes four random bits per question, one per feature.

### Independent Variables

1. **Encoding scheme** (4 stego conditions + honest)
2. **Number of observations N** (1, 2, 5, 10, 20, 50)
3. **Detector feature set**:
   - Narrow: only lexical + structural partitions (hand-crafted, domain-informed)
   - Broad: all families including learned partitions
   - Oracle: known codebook (upper bound)

### Metrics

1. **Detection AUC** (honest vs stego, across conditions) — the primary metric
2. **Scheme identification accuracy** — given detection fires, is the right family identified?
3. **False positive rate** at fixed detection thresholds
4. **AUC vs N curves** — how many observations needed for reliable detection?

### Key Experiments

**Experiment 1: Detection Power vs. Observations**

Plot AUC as a function of N for each scheme, comparing:
- Oracle detector (known codebook) — upper bound
- Broad feature detector (codebook-free, all partition families)
- Narrow feature detector (codebook-free, hand-crafted only)
- Perplexity baseline (is stego text less natural?)
- LLM judge baseline (prompt-based "does this look encoded?")
- Random baseline — lower bound

Expected result: oracle dominates at small N. Codebook-free detector converges toward oracle as N grows (the multi-observation signal compensates for not knowing the codebook). Perplexity baseline is flat and weak.

**Experiment 2: Generalization to Unseen Schemes**

The position scheme has no matching candidate partition. Does the detector still catch it via:
- Statistical partitions (position encoding might correlate with word-length features)?
- Learned partitions (embedding clusters)?

Expected result: reduced but nonzero AUC, demonstrating partial transfer.

**Experiment 3: Feature Ablation**

Remove one partition family at a time. Which features are necessary for detecting which schemes? This maps the relationship between partition families and encoding schemes — showing that the detector doesn't need the exact codebook, just correlated features.

**Experiment 4: Multiple Correction / False Discovery**

Testing M candidate partitions inflates false positives (multiple comparisons). Measure: what is the actual false positive rate of max_j S_j under honest agents? Apply Bonferroni or permutation-based correction. Report adjusted thresholds.

**Experiment 5: Bandwidth vs. Detectability**

Compare B=1 (single-scheme) vs B=4 (multi-scheme) conditions. At each N, plot AUC for both. Expected result: B=4 reaches high AUC at much smaller N because the conjunction of 4 independent partition biases is far less likely under honest behavior. This quantifies the attacker's bandwidth-detectability tradeoff.

**Experiment 6 (stretch): Approximate Distribution Preservation**

If the attacker uses best-of-k rejection sampling (generate k responses, pick the one that encodes the target bit AND is most natural), the steganography tax decreases with k. Plot detection AUC vs attacker's k. This traces the curve from "naive stego" toward "approximately secure stego" and shows where detection breaks down.

## What This Proves

1. **Inverse planning detects constraint, not content.** The detector works without the codebook because it searches for the *structural signature* of binary partitioning, not the specific encoding.

2. **Multi-observation compounding is the mechanism.** Single-observation detection is weak (too many candidate partitions, multiple comparison penalty). But across observations, the true partition accumulates signal while false partitions don't — because the attacker's constraint is consistent.

3. **The approach has a precise failure boundary.** It fails when the attacker perfectly preserves the output distribution (MEC). It partially fails when the attacker approximately preserves it (best-of-k). This is characterized, not hand-waved.

4. **Cost scales with reference samples, not codebook space.** The detector needs K reference samples per query (perhaps K = 50-100). It does NOT need to enumerate the combinatorial space of possible codebooks.

## What We Need

- [ ] AQUA-RAT questions (have this via HuggingFace loader)
- [ ] Honest agent responses (K per question, for reference distribution)
- [ ] Stego agent responses (one per question per scheme, encoding random bits)
- [ ] API access for generating responses (or use existing data + stub model for proof of concept)
- [ ] Implementation of candidate partition library
- [ ] Multiple comparison correction for max_j S_j

## Relationship to Existing Work

| Work | What they do | How we differ |
|---|---|---|
| Roger & Greenblatt (2023) | Paraphrasing as defense | We do statistical detection, not paraphrasing |
| Motwani et al. (2024) | Formalize collusion, evaluate capability | We propose a specific detection algorithm |
| Mathew et al. (2024) | Show stego emerges from RL, test monitors | We use inverse planning, not LLM judges |
| Bai et al. (2024) | Fine-tune classifiers on stego text | We are unsupervised (no stego training data needed) |
| Westphal et al. (2026) | Gray-box detection via linear probes | We are black-box (output-level only) |
| Schroeder de Witt et al. (2023) | Prove MEC is perfectly secure | We characterize where detection IS possible (non-MEC schemes) |

**The gap we fill:** No published work uses inverse planning for steganography detection. No published work detects covertext steganography in an unsupervised, codebook-free manner by searching for binary partition signatures across multiple observations.

## Open Questions

1. **How many candidate partitions can we test before multiple comparison correction kills the signal?** If M is too large, the threshold for detection becomes so high that only extremely biased schemes are catchable.

2. **Is the "learned partition" (embedding clustering) actually useful, or is it dominated by irrelevant variation?** The top axis of variation in response embeddings might be topic or length, not encoding-relevant features.

3. **How sensitive is detection to the reference model?** If reference responses come from a different model than the stego agent, does the distributional mismatch create false positives or mask true positives?

4. **Can an attacker who knows the detector's partition library explicitly avoid all tested partitions?** This is the adversarial robustness question. The learned partitions provide some defense (they're query-specific and data-driven), but a sufficiently sophisticated attacker could evade.


## Limitations
*The proposal's primary vulnerabilities stem from statistical thresholding limits and adversarial evasion tactics.*

### Statistical and Algorithmic Limits

* Testing a large library of candidate partitions requires multiple comparison corrections, which raises the detection threshold and risks masking the signal of subtly biased schemes.
* The data-driven "learned partitions," which use $k=2$ clustering on reference embeddings, may lack statistical power if the dominant axes of variation capture irrelevant features like topic or length rather than the steganographic encoding. Thinking more about this: clustering algorithm will greedily grab the most obvious axes of variation, completely ignoring the subtle steganographic boundary. maybe using PCA might be useful here. 
* The detector is sensitive to reference model mismatch; if the reference responses are sampled from a different model than the stego agent, the resulting distributional differences could trigger false positives or hide true positives.

### Adversarial and Theoretical Boundaries

* The detection approach completely fails against attackers who utilize encoding schemes that perfectly preserve the output distribution.
* Detection efficacy drops if the attacker employs best-of-$k$ rejection sampling to approximately preserve the natural output distribution, lowering the steganography tax.
* An attacker aware of the specific candidate partitions in the detector's library could explicitly design a scheme to evade them.

### Implementation Hurdles

* The system currently lacks an implementation for both the candidate partition library and the required multiple comparison correction logic.
