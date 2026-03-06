"""Unit tests for AUC computation."""

from src.eval.auc import compute_auc


def test_perfect_separation():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0]
    labels = [1, 1, 1, 0, 0, 0]
    assert compute_auc(scores, labels) == 1.0


def test_random_gives_half():
    # All same score -> AUC = 0.5
    scores = [0.5] * 10
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert compute_auc(scores, labels) == 0.5


def test_inverse_gives_zero():
    scores = [0.0, 0.1, 0.2, 0.7, 0.8, 0.9]
    labels = [1, 1, 1, 0, 0, 0]
    assert compute_auc(scores, labels) == 0.0


def test_single_class_gives_half():
    scores = [0.9, 0.5, 0.1]
    labels = [1, 1, 1]
    assert compute_auc(scores, labels) == 0.5
