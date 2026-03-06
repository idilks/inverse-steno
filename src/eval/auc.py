"""AUC computation from detection scores and labels."""


def compute_auc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC-ROC from scores and binary labels.

    Uses the Mann-Whitney U statistic formulation (no sklearn dependency).

    Args:
        scores: detection scores (higher = more suspicious)
        labels: 0 = honest, 1 = colluding

    Returns:
        AUC-ROC value in [0, 1]
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            # Trapezoidal rule
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_fp = fp
            prev_tp = tp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    auc += (fp - prev_fp) * (tp + prev_tp) / 2

    return auc / (n_pos * n_neg)
