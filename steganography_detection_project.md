# Inverse Planning for Steganography Detection

## Core Claim

Inverse planning detects covertext steganography by modeling task-optimal behavior and flagging deviations. Unlike grounding functions (which require enumerating what an agent *knows*), inverse planning requires enumerating what an agent *wants*. For covertext steganography, the objective space (task vs task+channel) is smaller and more tractable than the knowledge space (all possible encoding schemes).

## The Detection Problem

An agent performs a cover task (e.g., answer a math question). We suspect it may also be encoding a hidden message. We want to detect this.

**Two hypotheses:**
- H_honest: agent optimizes only for task performance
- H_collusion: agent optimizes for task performance AND hidden bit transfer

The colluding agent pays a "steganography tax": satisfying two objectives simultaneously forces suboptimal choices for the cover task. This suboptimality is the detection signal.

## Why This Matters

Perfectly secure steganography (via minimum entropy coupling) is information-theoretically undetectable. However, it requires:
1. Full logit access (not available in most deployed APIs)
2. Computational resources for MEC (GPT-4 can only compute MECs up to dim 2 without tools)
3. Shared randomness between sender/receiver (coordination is unreliable: ~16-22% success on complex schemes)
4. Identical computation reconstruction by receiver

Current LLMs cannot do this autonomously. Covertext modification steganography IS feasible and IS detectable. Our detector targets this realistic near-term threat.

## Formal Setup

### Honest Agent Model

Agent has task T. Follows Boltzmann policy:

```
P_honest(action | task) ∝ exp(β · Q_task(action))
```

where Q_task(action) measures action quality for the task.

### Colluding Agent Model

Agent has task T and secret bit b ∈ {0, 1} to encode.

```
P_collusion(action | task, bit) ∝ exp(β · Q_task(action)) · 𝟙[encode(action) = bit]
```

The indicator function forces the agent to only choose actions compatible with the bit.

### Marginalizing Over Unknown Bits

We don't know what bit the agent is encoding. Marginalize:

```
P_collusion(action | task) = 0.5 · P_collusion(action | task, bit=0) + 0.5 · P_collusion(action | task, bit=1)
```

### Marginalizing Over Unknown Schemes

We may not know the exact encoding scheme. Marginalize over a family:

```
P_collusion(action | task) = Σ_s P(scheme=s) · P_collusion(action | task, scheme=s)
```

### Detection Score

```
log_odds = log P(actions | H_collusion) - log P(actions | H_honest)
```

Higher log_odds → more likely colluding.

## Evaluation Framework

### Benchmark: Redwood Research Text-Steganography-Benchmark

Repository: https://github.com/redwoodresearch/Text-Steganography-Benchmark

**Why this benchmark:**
- Clean attacker/defender interfaces
- Established datasets (AQUA-RAT, Alpaca)
- Measures both capacity and task performance
- Open source, documented
- Already has baseline defenders to compare against

**Task structure:**
- Cover task: Answer math question (AQUA-RAT) or follow instruction (Alpaca)
- Hidden payload: Random bitstring (e.g., 4-8 bits)
- Encoder: Generates answer that solves task AND encodes bits
- Decoder: Extracts bits from answer
- Defender (us): Detects whether encoding occurred

### Steganography Schemes (from benchmark)

| Scheme | Type | Description |
|--------|------|-------------|
| Cycle-shifting | Watermarking | Shifts token distribution cyclically |
| Balance-marking | Watermarking | Marks based on token balance |
| Age instruction | Ad-hoc | Changes phrasing based on bit |
| Intro variation | Ad-hoc | Varies sentence structure |
| Parity encoding | Simple | Even/odd word count per sentence |

### Baseline Detectors

| Detector | Method |
|----------|--------|
| Statistical test | Chi-squared on output distribution |
| LLM judge | "Does this seem suspicious?" |
| Paraphraser | Rewrite and check if info survives |

### Our Detector

Inverse planning: compute P(output | task-only objective) vs P(output | task+channel objective). Flag if ratio exceeds threshold.

### Metrics

**Primary: AUC (Area Under ROC Curve)**
- Generate N samples: N/2 honest, N/2 steganographic
- Compute detector score for each
- AUC measures separation quality
- AUC = 0.5 → random guessing
- AUC = 1.0 → perfect detection

**Secondary:**
- FPR at fixed TPR (e.g., FPR when TPR=0.9)
- Calibration (does score correlate with actual collusion probability?)

### Experimental Conditions

**Independent variables:**
1. Encoding scheme complexity (simple parity → sophisticated watermarking)
2. Detector knowledge (knows exact scheme / knows family / knows nothing)
3. Number of observations (1 message / 10 messages / 100 messages)
4. Task difficulty (easy → hard math problems)

**Results table structure:**

| Scheme | Detector | Knowledge | N_obs | AUC |
|--------|----------|-----------|-------|-----|
| parity | inverse_planning | exact | 1 | ? |
| parity | inverse_planning | family | 1 | ? |
| parity | llm_judge | none | 1 | ? |
| ... | ... | ... | ... | ... |

## Implementation Sketch

### Task Model

```python
class MathTask:
    def __init__(self, question, correct_answer):
        self.question = question
        self.correct_answer = correct_answer
    
    def q_value(self, response):
        # Score based on correctness, clarity, conciseness
        # Higher = better for task
        ...
```

### Honest Policy

```python
def honest_log_prob(response, task, beta=1.0):
    q = task.q_value(response)
    # Would need to normalize over response space
    # In practice: use LLM log-probs as proxy
    return beta * q
```

### Collusion Policy

```python
def collusion_log_prob(response, task, encoding_scheme, beta=1.0):
    q = task.q_value(response)
    
    # Check which bit this response encodes
    bit = encoding_scheme(response)
    
    # Marginalize over unknown target bit
    # P(response) = 0.5 * P(response | bit=0) + 0.5 * P(response | bit=1)
    # If response encodes bit=b, then P(response|bit=b) = exp(βq)/Z_b
    # and P(response|bit=1-b) = 0
    
    return beta * q + log(0.5)  # simplified
```

### Detector

```python
def detect(response, task, encoding_schemes, beta=1.0):
    ll_honest = honest_log_prob(response, task, beta)
    
    ll_collusion_by_scheme = {}
    for name, scheme in encoding_schemes.items():
        ll_collusion_by_scheme[name] = collusion_log_prob(
            response, task, scheme, beta
        )
    
    best_scheme = max(ll_collusion_by_scheme, key=ll_collusion_by_scheme.get)
    ll_collusion = ll_collusion_by_scheme[best_scheme]
    
    log_odds = ll_collusion - ll_honest
    
    return {
        'log_odds': log_odds,
        'best_scheme': best_scheme,
        'll_honest': ll_honest,
        'll_collusion': ll_collusion_by_scheme
    }
```

### Evaluation Loop

```python
def run_evaluation(n_trials, task_generator, steganography_scheme, detector):
    trials = []
    
    for _ in range(n_trials // 2):
        task = task_generator()
        response = generate_honest_response(task)
        score = detector(response, task)
        trials.append({'label': 0, 'score': score['log_odds']})
    
    for _ in range(n_trials // 2):
        task = task_generator()
        secret_bits = random_bits(k=4)
        response = generate_stego_response(task, secret_bits, steganography_scheme)
        score = detector(response, task)
        trials.append({'label': 1, 'score': score['log_odds']})
    
    labels = [t['label'] for t in trials]
    scores = [t['score'] for t in trials]
    
    return roc_auc_score(labels, scores)
```

## Key Limitations

### Theoretical Limitations

1. **Requires knowing encoding family.** If agent uses unanticipated scheme, detection fails. We can marginalize over known schemes, but unknown unknowns defeat us.

2. **Perfectly secure steganography defeats this.** MEC-based encoding leaves no statistical trace by construction. This detector targets covertext modification only. This is acceptable because MEC requires capabilities current LLMs lack autonomously.

3. **Semantic steganography is hard to catch.** If encoding uses meaning rather than surface form (e.g., positive sentiment = 1), paraphrasing preserves the channel. Inverse planning helps only if semantic choices are suboptimal for the task.

### Practical Limitations

4. **Task model must be accurate.** Misspecified Q_task causes:
   - False positives: honest responses look suspicious
   - False negatives: colluding responses look normal
   
   For math tasks, correctness is clear. For open-ended tasks, "optimal" is fuzzy.

5. **Computing Q_task for natural language is hard.** Options:
   - Use LLM log-probs as proxy (assumes model knows what's natural)
   - Use separate evaluator model (adds complexity, potential bias)
   - Restrict to tasks with clear optima (multiple choice, numeric)

6. **Computational cost.** Must evaluate Q_task for each response. For large-scale deployment, need approximations.

7. **Baseline rate of suboptimality.** Honest agents aren't perfectly rational. Bounded rationality, typos, stylistic choices all create noise. Detection threshold must account for this.

### Scope Limitations (for this project)

8. **Single dataset.** Results may not generalize to other domains (code, creative writing, dialogue).

9. **Known encoding schemes.** We test against schemes we implement. Adversarial or emergent schemes untested.

10. **No actual multi-agent collusion.** We simulate sender encoding bits, but don't test full sender-receiver coordination with a real receiver agent.

## Project Scope (10 days, 3 people)

### In Scope

- 1 dataset: AQUA-RAT (math reasoning)
- 3 encoding schemes:
  - Simple parity (word count mod 2)
  - Position-based (first letter of Nth word)
  - One watermarking scheme from Redwood benchmark
- 2 baselines:
  - LLM judge ("does this response seem to hide information?")
  - Statistical test (chi-squared on surface features)
- 1 primary metric: AUC
- 1 secondary analysis: detection AUC vs number of observations

### Out of Scope (future work)

- Multiple datasets
- Semantic encoding schemes
- Full multi-agent coordination experiments
- Mechanistic interpretability probes
- Adversarial scheme optimization
- MEC-based steganography (undetectable anyway)

### Team Division (suggested)

| Person | Responsibility |
|--------|----------------|
| A | Data pipeline: AQUA-RAT loading, honest/stego response generation |
| B | Detector implementation: inverse planning model, baseline detectors |
| C | Evaluation: metrics, plotting, ablations, paper writing |

### Timeline (suggested)

| Days | Milestone |
|------|-----------|
| 1-2 | Setup: repo, data loading, basic response generation working |
| 3-4 | Encoding schemes implemented, honest vs stego samples generated |
| 5-6 | Detectors implemented (inverse planning + baselines) |
| 7-8 | Full evaluation runs, collect AUC numbers |
| 9-10 | Analysis, plots, write-up, presentation prep |

## Expected Results

**Where inverse planning should win:**
- Structured encodings (parity, position-based) where we can enumerate the scheme family
- Multiple observations from same agent (patterns emerge)
- Tasks with clear optimality criteria (math has right/wrong answers)

**Where inverse planning should lose:**
- Semantic encodings (positive/negative sentiment) that don't constrain surface form
- Single observations with high task variance
- Adversarially designed schemes that minimize detection

**The "detection frontier":**
- Level 1-2 schemes: AUC > 0.8 expected
- Level 3-4 schemes: AUC 0.6-0.8 expected  
- Level 5 (MEC): AUC ≈ 0.5 (undetectable by design)

## Resources

**Primary paper:**
- Motwani et al. (2024). "Secret Collusion among AI Agents: Multi-Agent Deception via Steganography." NeurIPS 2024.

**Benchmark:**
- Roger & Greenblatt (2023). "Preventing Language Models From Hiding Their Reasoning."
- GitHub: https://github.com/redwoodresearch/Text-Steganography-Benchmark

**Related work:**
- "Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs" (Mathew et al.)
- "Early Signs of Steganographic Capabilities in Frontier LLMs" (2025)

**Inverse planning reference:**
- Tan Xuan's Gen.jl notebook for inverse planning from scratch
https://colab.research.google.com/drive/11riF0x5X4tx-2qwWUthE72vKHdsx1KS5