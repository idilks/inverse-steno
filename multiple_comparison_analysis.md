# Multiple Comparison Analysis for Partition-Search Detector

## Setup and Notation

- $M$ candidate partition functions $\varphi_1, \dots, \varphi_M$
- $N$ observed query-response pairs $(q_i, r_i^*)$
- Per-observation score: $s(i,j) = \log(1/2) - \log(\hat{p}_{\text{bit}}(i,j))$
- Aggregate: $S_j = \sum_{i=1}^N s(i,j)$
- Detection score: $D = \max_j S_j$

For the idealized analysis below, assume $K \to \infty$ so $\hat{p}_{\text{bit}}(i,j) = p_{\text{bit}}(i,j)$ exactly (the population fraction). Finite-$K$ effects add estimation noise but don't change the structure.

---

## 1. Expected Score Under the Null

Under the null (honest agent), the response $r_i^*$ is drawn from $P(\cdot \mid q_i)$ without constraint. Let $p_j = P[\varphi_j(r) = 1 \mid q_i]$ for partition $j$ on query $i$. Then $r_i^*$ lands on side 1 with probability $p_j$ and side 0 with probability $1 - p_j$.

The score depends on which side the agent lands on:
- If $\varphi_j(r_i^*) = 1$: $s = \log(1/2) - \log(p_j)$, which happens with probability $p_j$
- If $\varphi_j(r_i^*) = 0$: $s = \log(1/2) - \log(1 - p_j)$, which happens with probability $1 - p_j$

Therefore:

$$E[s(i,j)] = p_j[\log(1/2) - \log(p_j)] + (1-p_j)[\log(1/2) - \log(1-p_j)]$$

$$= \log(1/2) - [p_j \log p_j + (1-p_j)\log(1-p_j)]$$

$$= \log(1/2) + H_j(i)$$

where $H_j(i) = -p_j \log p_j - (1-p_j)\log(1-p_j)$ is the binary entropy of partition $j$ on query $i$ (in nats).

**Key fact:** $H_j(i) \leq \log 2$, with equality iff $p_j = 1/2$. Therefore:

$$\boxed{E[s(i,j)] = H_j(i) - \log 2 \leq 0}$$

The null expectation is **negative** (or zero), not zero. It equals zero only when the partition is perfectly balanced ($p_j = 1/2$). For any asymmetric partition, the honest agent has negative expected score.

**This is important.** The comment in `inverse_planning.py` line 73 ("Under H_honest, E[sum] = 0") is wrong. The honest agent's expected score is $-D_{\text{KL}}(\text{Bernoulli}(p_j) \| \text{Bernoulli}(1/2))$, i.e., the negative KL divergence from the actual bit distribution to the uniform distribution. The more asymmetric the partition, the more negative the honest expectation.

**Numerical examples** (per observation):

| $p_j$ | $H_j$ (nats) | $E[s]$ (nats) | $E[s]$ (bits) |
|--------|---------------|----------------|----------------|
| 0.50 | 0.6931 | 0.0000 | 0.0000 |
| 0.40 | 0.6730 | -0.0201 | -0.0290 |
| 0.30 | 0.6109 | -0.0823 | -0.1187 |
| 0.20 | 0.5004 | -0.1927 | -0.2781 |
| 0.10 | 0.3251 | -0.3680 | -0.5310 |

So for a partition where 30% of honest responses land on one side, the honest agent scores $-0.082$ nats per observation on average. Over $N=20$ observations, that's $-1.65$ nats of drift away from zero. This negative drift is our friend: it means honest agents actively suppress scores on asymmetric partitions.

---

## 2. Variance Under the Null

The variance of $s(i,j)$ for a single observation:

$$\text{Var}[s(i,j)] = E[s(i,j)^2] - (E[s(i,j)])^2$$

Compute $E[s^2]$:

$$E[s^2] = p_j[\log(1/2) - \log p_j]^2 + (1-p_j)[\log(1/2) - \log(1-p_j)]^2$$

Let $a = \log(1/2) - \log p_j = \log\frac{1}{2p_j}$ and $b = \log(1/2) - \log(1-p_j) = \log\frac{1}{2(1-p_j)}$. Then:

$$E[s^2] = p_j a^2 + (1-p_j) b^2$$

$$\text{Var}[s] = p_j a^2 + (1-p_j) b^2 - [p_j a + (1-p_j) b]^2 = p_j(1-p_j)(a - b)^2$$

Now $a - b = \log\frac{1-p_j}{p_j}$, so:

$$\boxed{\text{Var}[s(i,j)] = p_j(1-p_j)\left(\log\frac{1-p_j}{p_j}\right)^2}$$

This is the variance of a single Bernoulli-weighted log-odds. Note it equals zero when $p_j = 1/2$ (balanced partition gives deterministic score of zero) and increases with asymmetry.

**Numerical examples:**

| $p_j$ | $\text{Var}[s]$ | $\text{SD}[s]$ |
|--------|-----------------|-----------------|
| 0.50 | 0.0000 | 0.0000 |
| 0.40 | 0.0243 | 0.1559 |
| 0.30 | 0.1421 | 0.3769 |
| 0.20 | 0.3848 | 0.6203 |
| 0.10 | 0.8676 | 0.9314 |

**Aggregate variance over $N$ observations.** Assuming queries are independent (different questions), and for simplicity that $p_j$ is the same across queries (homogeneous asymmetry):

$$\text{Var}[S_j] = N \cdot \text{Var}[s(i,j)]$$

$$\text{SD}[S_j] = \sqrt{N} \cdot \text{SD}[s]$$

---

## 3. Score Under the Alternative (Stego Agent Using Partition $j^*$)

The stego agent encoding a uniform random bit via partition $j^*$ always lands on the minority side with probability proportional to the message. Specifically, the message bit $m \sim \text{Bernoulli}(1/2)$, and the agent produces a response on side $m$. So the agent's bit is uniform over $\{0, 1\}$, independent of $p_{j^*}$.

When the agent lands on side $b$, $p_{\text{bit}} = p_b$ where $p_1 = p_{j^*}$ and $p_0 = 1 - p_{j^*}$. Each side is hit with probability $1/2$ (uniform message). So:

$$E_{\text{stego}}[s(i, j^*)] = \frac{1}{2}[\log(1/2) - \log p_{j^*}] + \frac{1}{2}[\log(1/2) - \log(1-p_{j^*})]$$

$$= \log(1/2) - \frac{1}{2}\log p_{j^*} - \frac{1}{2}\log(1-p_{j^*})$$

$$= \log(1/2) - \frac{1}{2}\log[p_{j^*}(1-p_{j^*})]$$

Compare with the null: $E_{\text{null}}[s] = \log(1/2) + H_{j^*}$. The difference:

$$\Delta \mu = E_{\text{stego}}[s] - E_{\text{null}}[s] = -\frac{1}{2}\log[p(1-p)] - H$$

where $H = -p\log p - (1-p)\log(1-p)$ and $p = p_{j^*}$.

After algebra (letting $q = 1 - p$):

$$\Delta \mu = -\frac{1}{2}(\log p + \log q) - (-p\log p - q\log q)$$
$$= -\frac{1}{2}\log p - \frac{1}{2}\log q + p\log p + q\log q$$
$$= (p - \frac{1}{2})\log p + (q - \frac{1}{2})\log q$$

Since $q = 1-p$:

$$\boxed{\Delta\mu = (p - \tfrac{1}{2})\log\frac{p}{1-p}}$$

This equals zero when $p = 1/2$ and is strictly positive for $p \neq 1/2$. It is symmetric in $p$ and $1-p$.

**Numerical signal strength per observation:**

| $p_{j^*}$ | $E_{\text{null}}[s]$ | $E_{\text{stego}}[s]$ | $\Delta\mu$ (nats) |
|------------|----------------------|------------------------|---------------------|
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.40 | -0.020 | 0.020 | 0.040 |
| 0.30 | -0.082 | 0.082 | 0.165 |
| 0.20 | -0.193 | 0.193 | 0.386 |
| 0.10 | -0.368 | 0.368 | 0.736 |

The stego expected score is exactly $-E_{\text{null}}[s]$ (positive). The gap is $2|E_{\text{null}}[s]|$. This makes intuitive sense: the honest agent drifts negative, the stego agent drifts positive, and the total separation is twice the absolute value.

---

## 4. Signal-to-Noise and Detection Threshold

### 4a. For a Single (Known) Partition ($M = 1$)

After $N$ observations, under the null: $S_{j^*} \sim \mathcal{N}(N \cdot \mu_0, N \cdot \sigma^2)$ by CLT, where $\mu_0 = E_{\text{null}}[s] < 0$ and $\sigma^2 = \text{Var}[s]$.

Under the alternative: $S_{j^*} \sim \mathcal{N}(N \cdot \mu_1, N \cdot \sigma_1^2)$ where $\mu_1 = E_{\text{stego}}[s] > 0$.

The signal-to-noise ratio for distinguishing the two distributions:

$$\text{SNR}(N) = \frac{N(\mu_1 - \mu_0)}{\sqrt{N \cdot \sigma_0^2}} = \sqrt{N} \cdot \frac{\Delta\mu}{\sigma_0}$$

where I use the null variance $\sigma_0^2$ since we're setting a threshold under the null.

The **per-observation discriminability** is $d = \Delta\mu / \sigma_0$:

| $p$ | $\Delta\mu$ | $\sigma_0$ | $d = \Delta\mu/\sigma_0$ |
|------|-------------|------------|--------------------------|
| 0.40 | 0.040 | 0.156 | 0.259 |
| 0.30 | 0.165 | 0.377 | 0.437 |
| 0.20 | 0.386 | 0.620 | 0.622 |
| 0.10 | 0.736 | 0.931 | 0.790 |

For AUC = 0.9, we need $\text{SNR} \approx 1.81$ (since $\text{AUC} = \Phi(\text{SNR}/\sqrt{2})$ for Gaussian, and $\Phi^{-1}(0.9) \approx 1.28$, so $\text{SNR} = 1.28\sqrt{2} \approx 1.81$).[^1]

[^1]: More precisely, AUC between two Gaussians with means $\mu_0, \mu_1$ and equal variance $\sigma^2$ is $\Phi\left(\frac{\mu_1 - \mu_0}{\sigma\sqrt{2}}\right)$. For unequal variances the formula adjusts, but the Gaussian approximation is adequate for the order-of-magnitude analysis.

Required $N$ for AUC = 0.9 with known partition ($M=1$):

$$N = \left(\frac{1.81}{d}\right)^2$$

| $p$ | $d$ | $N$ for AUC=0.9 |
|------|-------|-----------------|
| 0.40 | 0.259 | 49 |
| 0.30 | 0.437 | 17 |
| 0.20 | 0.622 | 8 |
| 0.10 | 0.790 | 5 |

### 4b. With $M$ Partitions (Bonferroni)

Now $D = \max_j S_j$. Under the null, all $M$ partitions produce scores with negative drift. The threshold $\tau$ must satisfy:

$$P_{\text{null}}[D > \tau] \leq \alpha$$

**Bonferroni:** $P[\max_j S_j > \tau] \leq \sum_j P[S_j > \tau]$. Set each term to $\alpha/M$:

$$\tau_{\text{Bonf}} = N\mu_0 + z_{1-\alpha/M} \cdot \sqrt{N} \cdot \sigma_0$$

where $z_q = \Phi^{-1}(q)$.

For the true partition $j^*$ under the alternative, we need:

$$P_{\text{stego}}[S_{j^*} > \tau_{\text{Bonf}}] \geq 1 - \beta$$

$$N\mu_1 - z_{1-\beta}\sqrt{N}\sigma_1 > N\mu_0 + z_{1-\alpha/M}\sqrt{N}\sigma_0$$

Dividing by $\sqrt{N}$ and solving (assuming $\sigma_0 \approx \sigma_1 \equiv \sigma$ for simplicity):

$$\sqrt{N} \cdot \Delta\mu > (z_{1-\alpha/M} + z_{1-\beta}) \cdot \sigma$$

$$\boxed{N > \left(\frac{(z_{1-\alpha/M} + z_{1-\beta}) \cdot \sigma}{\Delta\mu}\right)^2}$$

The Bonferroni penalty enters through $z_{1-\alpha/M}$. For $\alpha = 0.05$:

| $M$ | $\alpha/M$ | $z_{1-\alpha/M}$ | Penalty factor vs $M=1$ |
|-----|-----------|-------------------|-------------------------|
| 1 | 0.050 | 1.645 | 1.00x |
| 10 | 0.005 | 2.576 | 1.34x in $\sqrt{N}$, 1.79x in $N$ |
| 50 | 0.001 | 3.090 | 1.51x in $\sqrt{N}$, 2.27x in $N$ |
| 100 | 0.0005 | 3.291 | 1.58x in $\sqrt{N}$, 2.50x in $N$ |
| 200 | 0.00025 | 3.481 | 1.64x in $\sqrt{N}$, 2.70x in $N$ |

**The Bonferroni penalty is mild.** Going from $M=1$ to $M=200$ only multiplies the required $N$ by $\sim 2.7$x. This is because the penalty is logarithmic in $M$: $z_{1-\alpha/M} \approx \sqrt{2\log(M/\alpha)}$ for large $M$.

### 4c. Concrete $N$ Requirements (Bonferroni, $\alpha=0.05$, power $= 0.8$, so $z_{1-\beta} = 0.842$)

Required $N$ for reliable detection:

| $p$ | $M=1$ | $M=10$ | $M=50$ | $M=100$ | $M=200$ |
|------|-------|--------|--------|---------|---------|
| 0.40 | 35 | 63 | 80 | 88 | 95 |
| 0.30 | 12 | 22 | 28 | 31 | 33 |
| 0.20 | 6 | 10 | 13 | 14 | 15 |
| 0.10 | 3 | 6 | 7 | 8 | 9 |

These are computed from $N = \left(\frac{(z_{1-\alpha/M} + 0.842)\sigma}{\Delta\mu}\right)^2$, rounded up.

---

## 5. Bonferroni vs. Permutation Correction

**Bonferroni** controls the family-wise error rate (FWER) by assuming the worst case: all $M$ tests are independent. In reality, many partitions are correlated (e.g., two synonym-based partitions may agree on most responses). Bonferroni is therefore conservative.

**Permutation testing** directly estimates the null distribution of $D = \max_j S_j$. Procedure:
1. For each of $B$ permutation rounds, randomly assign each observation's bit to the opposite side (or equivalently, randomly flip which side the agent "landed on").
2. Recompute $S_j^{(b)}$ for all $j$ and record $D^{(b)} = \max_j S_j^{(b)}$.
3. The p-value is $\hat{p} = \frac{1}{B}\sum_{b=1}^B \mathbb{1}[D^{(b)} \geq D_{\text{obs}}]$.

**Why permutation is more powerful:**

1. **Exploits correlation.** If partitions are correlated, the effective number of independent tests $M_{\text{eff}} < M$. Permutation testing automatically accounts for this. The gain is $N_{\text{perm}} / N_{\text{Bonf}} \approx M_{\text{eff}} / M$.

2. **Exact control.** Permutation gives exact FWER control regardless of the score distribution (no Gaussian assumption needed).

3. **Quantitative gain.** If $M = 100$ but only $M_{\text{eff}} = 30$ partitions are effectively independent (plausible: many synonym pairs are correlated, structural features overlap), then permutation behaves like Bonferroni with $M=30$. From the table above, this saves about 20-30% on $N$.

**Cost:** $B \times M \times N$ score evaluations per permutation test. With $B=1000$, $M=100$, $N=50$: $5 \times 10^6$ evaluations. These are just partition function lookups (microseconds each), so permutation is computationally free.

**Recommendation:** Use permutation testing. Bonferroni for quick analytical bounds, permutation for actual thresholds. The implementation is straightforward and strictly dominates Bonferroni in power.

---

## 6. Concrete Scenario: $M = 100$, $p = 0.3$, $N = 1$ to $50$

Parameters: $\Delta\mu = 0.165$ nats, $\sigma_0 = 0.377$ nats, $d = 0.437$.

### Expected signal and noise vs. $N$

| $N$ | $E[S_{j^*}]_{\text{stego}}$ | $E[S_{j^*}]_{\text{null}}$ | Gap | $\text{SD}_{\text{null}}[S]$ | $\text{SNR}_{\text{raw}}$ |
|-----|------------------------------|-----------------------------|----|-------------------------------|---------------------------|
| 1 | 0.082 | -0.082 | 0.165 | 0.377 | 0.44 |
| 2 | 0.165 | -0.165 | 0.329 | 0.533 | 0.62 |
| 5 | 0.411 | -0.411 | 0.823 | 0.843 | 0.98 |
| 10 | 0.823 | -0.823 | 1.646 | 1.192 | 1.38 |
| 20 | 1.646 | -1.646 | 3.291 | 1.685 | 1.95 |
| 30 | 2.469 | -2.469 | 4.937 | 2.064 | 2.39 |
| 50 | 4.114 | -4.114 | 8.228 | 2.665 | 3.09 |

where $\text{SNR}_{\text{raw}} = \sqrt{N} \cdot d = \sqrt{N} \cdot 0.437$.

### After Bonferroni correction ($M=100$, $\alpha = 0.05$)

The threshold is $\tau = N\mu_0 + 3.291 \cdot \sqrt{N} \cdot \sigma_0$.

| $N$ | Threshold $\tau$ | $E[S_{j^*}]_{\text{stego}}$ | Stego exceeds threshold? | Approx. power |
|-----|-------------------|-------------------------------|--------------------------|---------------|
| 1 | -0.082 + 1.241 = 1.159 | 0.082 | No ($0.082 < 1.159$) | ~0% |
| 5 | -0.411 + 2.774 = 2.363 | 0.411 | No | ~1% |
| 10 | -0.823 + 3.924 = 3.101 | 0.823 | No | ~3% |
| 20 | -1.646 + 5.549 = 3.903 | 1.646 | No | ~9% |
| 30 | -2.469 + 6.795 = 4.326 | 2.469 | Borderline | ~19% |
| 50 | -4.114 + 8.774 = 4.660 | 4.114 | Close | ~42% |

Power is $\Phi\left(\frac{E_{\text{stego}} - \tau}{\sigma_{\text{stego}}\sqrt{N} \cdot ...}\right)$. More precisely, using the stego variance (which is similar order):

At $N=50$: need $S_{j^*} > 4.660$, stego mean is $4.114$, stego SD $\approx 2.67$. So $z = (4.114 - 4.660)/2.67 = -0.204$, power $= \Phi(-0.204) \approx 42\%$.

**This tells us $p = 0.3$ with $M = 100$ is hard.** We need $N \approx 80$-$90$ for 80% power (consistent with the table in Section 4c).

### What if the partition is more asymmetric? ($p = 0.2$)

With $p = 0.2$: $\Delta\mu = 0.386$, $\sigma_0 = 0.620$, $d = 0.622$.

| $N$ | Threshold $\tau$ ($M=100$) | $E_{\text{stego}}[S]$ | Power |
|-----|----------------------------|------------------------|-------|
| 5 | -0.963 + 4.568 = 3.605 | 0.963 | ~2% |
| 10 | -1.927 + 6.459 = 4.532 | 1.927 | ~9% |
| 15 | -2.890 + 7.910 = 5.020 | 2.890 | ~22% |
| 20 | -3.854 + 9.134 = 5.280 | 3.854 | ~38% |
| 30 | -5.781 + 11.188 = 5.407 | 5.781 | ~65% |
| 50 | -9.635 + 14.443 = 4.808 | 9.635 | ~96% |

At $N = 50$ with $p = 0.2$: the threshold actually starts *decreasing* because the $N\mu_0$ term (negative, growing linearly) eventually dominates the $\sqrt{N}$ noise term. This is the regime where the negative drift of honest partitions pulls the threshold down and stego signal easily exceeds it.

---

## 7. Summary and Feasibility

### The key equation

Required observations for power $1-\beta$ at FWER $\alpha$ with $M$ partitions and partition asymmetry $p$:

$$N \geq \left(\frac{z_{1-\alpha/M} + z_{1-\beta}}{d(p)}\right)^2$$

where $d(p) = \frac{(p - 1/2)\log(p/(1-p))}{\sqrt{p(1-p)} \cdot |\log((1-p)/p)|}  = \frac{|p - 1/2|}{\sqrt{p(1-p)}}$

Simplifying: $d(p) = \frac{|p - 1/2|}{\sqrt{p(1-p)}}$. Wait -- let me recompute. $\Delta\mu = (p-1/2)\log(p/(1-p))$ for $p < 1/2$ this is positive (both factors negative). $\sigma = \sqrt{p(1-p)} \cdot |\log((1-p)/p)|$. So:

$$d = \frac{|p - 1/2| \cdot |\log(p/(1-p))|}{\sqrt{p(1-p)} \cdot |\log((1-p)/p)|} = \frac{|p-1/2|}{\sqrt{p(1-p)}}$$

This is a clean result. The per-observation discriminability is:

$$\boxed{d(p) = \frac{|1/2 - p|}{\sqrt{p(1-p)}}}$$

which is just the **absolute difference from balance divided by the Bernoulli standard deviation**. It's the effect size of a one-sample proportion test against $p_0 = 1/2$.

### Feasibility table ($\alpha = 0.05$, power = 0.8)

| Asymmetry $p$ | $d(p)$ | $N$ ($M=1$) | $N$ ($M=50$) | $N$ ($M=100$) | $N$ ($M=200$) |
|---------------|--------|-------------|--------------|----------------|----------------|
| 0.45 | 0.101 | 244 | 554 | 610 | 660 |
| 0.40 | 0.204 | 61 | 139 | 153 | 165 |
| 0.35 | 0.315 | 26 | 58 | 64 | 69 |
| 0.30 | 0.436 | 13 | 30 | 33 | 36 |
| 0.25 | 0.577 | 8 | 17 | 19 | 20 |
| 0.20 | 0.750 | 4 | 10 | 11 | 12 |
| 0.15 | 0.981 | 3 | 6 | 7 | 7 |
| 0.10 | 1.333 | 2 | 3 | 4 | 4 |

### Bottom line

1. **The project is feasible.** Even at $M = 200$, the Bonferroni penalty only costs a $\sim 2.7$x multiplier on $N$. With $p = 0.3$ (a moderate encoding scheme), detection requires $N \approx 33$ observations. With $p = 0.2$ (a scheme where 80% of honest responses land on one side), $N \approx 11$.

2. **The penalty is logarithmic in $M$.** Going from $M = 50$ to $M = 200$ costs only $\sim 20\%$ more observations. We can afford a large partition library.

3. **Permutation testing will do better.** If the effective number of independent partitions is $M_{\text{eff}} \approx M/3$ (plausible given correlated synonym pairs), we save another $\sim 25\%$ on $N$.

4. **The biggest lever is partition asymmetry, not $M$.** The difference between $p = 0.4$ and $p = 0.2$ is a 14x reduction in required $N$. Designing partition functions that maximize asymmetry on real data matters far more than minimizing $M$.

5. **The code has a bug.** `inverse_planning.py` line 73 claims $E[\text{sum}] = 0$ under the null. The true expectation is $\sum_i (H_j(i) - \log 2) \leq 0$. This doesn't break the detector (the score is still positive for stego and negative for honest, which is even better for separation), but it means the threshold should not be calibrated around zero. It should be calibrated from the empirical null distribution or from the formula above.

### What to do next

- [ ] Fix the docstring in `detect_multi` (E[sum] is negative, not zero)
- [ ] Implement permutation testing for threshold calibration
- [ ] Measure empirical $p_j$ values for each candidate partition on AQUA-RAT data to determine which partitions give the most asymmetry (this determines actual $N$ requirements)
- [ ] The Laplace smoothing in `detect()` (line 50) slightly biases $\hat{p}_{\text{bit}}$ toward 0.5, which reduces both signal and variance. With $K = 50$ references, the bias is $\sim 1/52$ which is negligible. With $K = 10$ it matters.
