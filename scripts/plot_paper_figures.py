"""Generate publication-quality figures for AAAI paper.

Figures:
1. AUC vs N curve (main result) — oracle, codebook-free, counting, embedding
2. Score accumulation schematic — linear drift vs random walk
3. Example interaction diagram (exported as standalone)

Usage:
    python scripts/plot_paper_figures.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_CF = ROOT / "results" / "diagnostic_embedding_cf.json"
RESULTS_ORACLE = ROOT / "results" / "diagnostic_oracle.json"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COLORS = {
    "oracle": "#2c3e50",
    "logodds": "#e74c3c",
    "counting": "#3498db",
    "embedding": "#95a5a6",
}
MARKERS = {"oracle": "s", "logodds": "o", "counting": "^", "embedding": "D"}


def load_results():
    """Load CF and oracle result files."""
    with open(RESULTS_CF) as f:
        cf = json.load(f)
    with open(RESULTS_ORACLE) as f:
        oracle = json.load(f)
    return cf, oracle


def extract_auc_vs_n(data, key="auc_mean"):
    """Extract {scheme: {N: value}} from aggregated_multi_obs."""
    multi = data["aggregated_multi_obs"]
    schemes = ["structural", "synonym", "parity"]
    result = {s: {} for s in schemes}
    for n_str, scheme_data in multi.items():
        n = int(n_str)
        for s in schemes:
            if s in scheme_data and scheme_data[s].get(key) is not None:
                result[s][n] = scheme_data[s][key]
    return result


def extract_auc_vs_n_std(data, key="auc_std"):
    """Extract std values."""
    return extract_auc_vs_n(data, key)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: AUC vs N (main result)
# ═══════════════════════════════════════════════════════════════════════════

def plot_auc_vs_n():
    cf, oracle_data = load_results()

    logodds = extract_auc_vs_n(cf, "auc_mean")
    logodds_std = extract_auc_vs_n(cf, "auc_std")
    counting = extract_auc_vs_n(cf, "counting_auc_mean")
    counting_std = extract_auc_vs_n(cf, "counting_auc_std")
    embedding = extract_auc_vs_n(cf, "embedding_auc_mean")
    embedding_std = extract_auc_vs_n(cf, "embedding_auc_std")
    orc = extract_auc_vs_n(oracle_data, "auc_mean")
    orc_std = extract_auc_vs_n(oracle_data, "auc_std")

    schemes = ["structural", "synonym", "parity"]
    titles = ["Structural", "Synonym", "Parity (control)"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)

    for ax, scheme, title in zip(axes, schemes, titles):
        for label, data, std_data, color, marker, ls in [
            ("Oracle", orc, orc_std, COLORS["oracle"], MARKERS["oracle"], "--"),
            ("Log-odds (CF)", logodds, logodds_std, COLORS["logodds"], MARKERS["logodds"], "-"),
            ("Counting (CF)", counting, counting_std, COLORS["counting"], MARKERS["counting"], "-"),
            ("Embedding (CF)", embedding, embedding_std, COLORS["embedding"], MARKERS["embedding"], ":"),
        ]:
            ns = sorted(data[scheme].keys())
            ys = [data[scheme][n] for n in ns]
            es = [std_data[scheme].get(n, 0) for n in ns] if std_data else [0]*len(ns)

            ax.plot(ns, ys, color=color, marker=marker, linestyle=ls,
                    label=label, zorder=3)
            ax.fill_between(ns, [y-e for y,e in zip(ys,es)],
                           [y+e for y,e in zip(ys,es)],
                           color=color, alpha=0.1, zorder=2)

        ax.set_title(title)
        ax.set_xlabel("Observations ($N$)")
        ax.set_xticks([1, 2, 3, 5, 7, 10])
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", zorder=1)
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("AUC")
    axes[2].legend(loc="lower right", framealpha=0.9)

    fig.tight_layout(w_pad=0.5)
    out = OUT_DIR / "auc_vs_n.pdf"
    fig.savefig(out)
    fig.savefig(OUT_DIR / "auc_vs_n.png")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Score accumulation schematic
# ═══════════════════════════════════════════════════════════════════════════

def plot_score_accumulation():
    """Stylized illustration: true partition drifts up, noise random-walks."""
    rng = np.random.RandomState(42)
    N = 15

    # True partition: positive drift
    true_scores = rng.normal(0.6, 0.3, N)
    true_cumsum = np.cumsum(true_scores)

    # Noise partitions: zero-mean random walks
    n_noise = 8
    noise_walks = []
    for _ in range(n_noise):
        steps = rng.normal(0.0, 0.4, N)
        noise_walks.append(np.cumsum(steps))

    fig, ax = plt.subplots(figsize=(3.3, 2.0))

    for walk in noise_walks:
        ax.plot(range(1, N+1), walk, color=COLORS["embedding"],
                alpha=0.4, linewidth=0.8, zorder=2)

    # Label one noise line
    ax.plot([], [], color=COLORS["embedding"], alpha=0.6, linewidth=0.8,
            label="Incorrect partitions")

    ax.plot(range(1, N+1), true_cumsum, color=COLORS["logodds"],
            linewidth=2.0, marker="o", markersize=3, zorder=4,
            label="True partition")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Observations ($N$)")
    ax.set_ylabel("Cumulative score $\\sum_i s(i, j)$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0.5, N + 0.5)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Annotate
    ax.annotate("$O(N)$ drift", xy=(N, true_cumsum[-1]),
                xytext=(N-4, true_cumsum[-1]+1.5),
                fontsize=7, color=COLORS["logodds"],
                arrowprops=dict(arrowstyle="->", color=COLORS["logodds"],
                               lw=0.8))
    # Point at the envelope of noise walks
    noise_max = max(w[-1] for w in noise_walks)
    noise_min = min(w[-1] for w in noise_walks)
    noise_mid = (noise_max + noise_min) / 2
    ax.annotate("$O(\\sqrt{N})$", xy=(N, noise_mid),
                xytext=(N-5, noise_mid - 2.5),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555",
                               lw=0.8))

    fig.tight_layout()
    out = OUT_DIR / "score_accumulation.pdf"
    fig.savefig(out)
    fig.savefig(OUT_DIR / "score_accumulation.png")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Example interaction
# ═══════════════════════════════════════════════════════════════════════════

def plot_example_interaction():
    """Visual example of how the detector scores a single observation.

    Compact layout for AAAI column width (~3.3in). Shows question, two
    responses with highlighted synonym, partition scoring.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fs_label = 6.5
    fs_body = 5.8
    fs_math = 6.2

    y = 0.96

    # ── Question ───────────────────────────────────────────────────────
    ax.text(0.0, y, "Q:", fontsize=fs_label, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.06, y,
            '"A store sells an item at 20% profit. If the cost\n  price is $150, what is the selling price?"',
            fontsize=fs_body, style="italic", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f7f7f7",
                      edgecolor="#cccccc", linewidth=0.4))

    y -= 0.17

    # ── Responses side-by-side ─────────────────────────────────────────
    ax.text(0.0, y, "Honest:", fontsize=fs_label, fontweight="bold",
            color="#2c3e50", transform=ax.transAxes, va="top")
    ax.text(0.5, y, "Stego:", fontsize=fs_label, fontweight="bold",
            color="#c0392b", transform=ax.transAxes, va="top")

    y -= 0.06
    ax.text(0.01, y,
            'Profit = 150 * 0.20 = 30.\nTherefore, selling price\n= 150 + 30 = $180. Answer: (C)',
            fontsize=fs_body, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#2c3e50", linewidth=0.6),
            family="monospace")
    ax.text(0.51, y,
            'Profit = 150 * 0.20 = 30.\nThus, selling price\n= 150 + 30 = $180. Answer: (C)',
            fontsize=fs_body, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5",
                      edgecolor="#c0392b", linewidth=0.6),
            family="monospace")

    y -= 0.19

    # ── Separator ──────────────────────────────────────────────────────
    ax.plot([0.02, 0.98], [y, y], color="#dddddd", linewidth=0.5,
            transform=ax.transAxes, clip_on=False)
    y -= 0.06

    # ── Scoring ────────────────────────────────────────────────────────
    ax.text(0.0, y,
            '$\\phi$: synonym pair  ("therefore" $\\to$ 0,  "thus" $\\to$ 1)',
            fontsize=fs_math, transform=ax.transAxes, va="top")
    y -= 0.08
    ax.text(0.0, y,
            'References: 17/20 use "therefore",  3/20 use "thus"',
            fontsize=fs_math, transform=ax.transAxes, va="top")
    y -= 0.08
    ax.text(0.0, y,
            'Observed: "thus" (bit 1)  $\\Rightarrow$  '
            '$\\hat{p}_{\\mathrm{bit}} = 3/20 = 0.15$',
            fontsize=fs_math, color="#c0392b",
            transform=ax.transAxes, va="top")
    y -= 0.09
    ax.text(0.0, y,
            '$s = \\log(0.5) - \\log(0.15) = 1.20$',
            fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.58, y, '(high score $\\to$ suspicious)',
            fontsize=fs_body, color="#888888",
            transform=ax.transAxes, va="top")

    out = OUT_DIR / "example_interaction.pdf"
    fig.savefig(out)
    fig.savefig(OUT_DIR / "example_interaction.png")
    print(f"Saved: {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating paper figures...")
    plot_auc_vs_n()
    plot_score_accumulation()
    plot_example_interaction()
    print(f"\nAll figures saved to {OUT_DIR}/")
