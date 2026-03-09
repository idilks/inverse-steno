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


def compute_fpr_at_tpr(
    scores: list[float], labels: list[int], target_tpr: float
) -> float:
    """Compute the FPR required to achieve a given TPR.

    Sweeps thresholds from high to low, returns the FPR at the first
    threshold where TPR >= target_tpr.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target_tpr:
            return fp / n_neg

    return 1.0


def compute_partial_auc(
    scores: list[float], labels: list[int], max_fpr: float
) -> float:
    """Compute partial AUC up to max_fpr, normalized to [0, 1].

    pAUC is the area under the ROC curve for FPR in [0, max_fpr],
    divided by max_fpr so the result is comparable to full AUC.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    prev_score = None
    pauc = 0.0

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            fpr = fp / n_neg
            if fpr > max_fpr:
                # Interpolate to max_fpr boundary
                prev_fpr = prev_fp / n_neg
                frac = (max_fpr - prev_fpr) / (fpr - prev_fpr)
                interp_tp = prev_tp + frac * (tp - prev_tp)
                pauc += (max_fpr - prev_fpr) * (interp_tp + prev_tp) / 2
                break
            pauc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_fp = fp
            prev_tp = tp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    else:
        fpr = fp / n_neg
        if fpr <= max_fpr:
            pauc += (fp - prev_fp) * (tp + prev_tp) / 2

    return pauc / (n_pos * n_neg * max_fpr) if max_fpr > 0 else 0.5
