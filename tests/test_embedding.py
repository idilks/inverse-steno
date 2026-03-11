"""Tests for embedding-based anomaly detection."""

import numpy as np
import pytest

from src.detector.embedding import (
    tfidf_embed,
    anomaly_score,
    detect_embedding,
    detect_embedding_multi,
    _char_ngrams,
    _svd_reduce,
    _loo_pvalue,
)


def _st_available():
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


class TestTfidfEmbed:
    def test_shape(self):
        texts = ["hello world", "foo bar baz", "another sentence here"]
        embs = tfidf_embed(texts, n=3, max_features=100)
        assert embs.shape[0] == 3
        assert embs.shape[1] <= 100

    def test_l2_normalized(self):
        texts = ["hello world foo", "bar baz quux thing"]
        embs = tfidf_embed(texts)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_identical_texts_identical_embeddings(self):
        texts = ["same text here", "same text here", "different one"]
        embs = tfidf_embed(texts)
        np.testing.assert_allclose(embs[0], embs[1])
        assert not np.allclose(embs[0], embs[2])

    def test_empty_text_handled(self):
        texts = ["", "ab", "abc def"]
        embs = tfidf_embed(texts, n=3)
        assert embs.shape[0] == 3


class TestCharNgrams:
    def test_basic(self):
        grams = _char_ngrams("hello", 3)
        assert grams == ["hel", "ell", "llo"]

    def test_short_text(self):
        assert _char_ngrams("ab", 3) == []


class TestSvdReduce:
    def test_reduces_dimensions(self):
        X = np.random.RandomState(42).randn(20, 50)
        reduced = _svd_reduce(X, n_components=10)
        assert reduced.shape == (20, 10)

    def test_noop_when_already_small(self):
        X = np.random.RandomState(42).randn(20, 5)
        reduced = _svd_reduce(X, n_components=10)
        assert reduced.shape == (20, 5)


class TestAnomalyScore:
    def test_outlier_scores_higher_than_inlier(self):
        """An obvious outlier should score higher than an inlier."""
        rng = np.random.RandomState(42)
        # Cluster at origin
        references = rng.randn(50, 10) * 0.1
        inlier = rng.randn(10) * 0.1
        outlier = np.ones(10) * 5.0  # far from cluster

        for method in ["knn", "mahalanobis", "cosine"]:
            score_in = anomaly_score(inlier, references, method=method)
            score_out = anomaly_score(outlier, references, method=method)
            assert score_out > score_in, f"{method}: outlier={score_out}, inlier={score_in}"

    def test_knn_with_few_references(self):
        """kNN should work even with very few references."""
        rng = np.random.RandomState(42)
        refs = rng.randn(5, 10)
        suspect = rng.randn(10)
        score = anomaly_score(suspect, refs, method="knn", k=3)
        assert isinstance(score, float)
        assert score > 0


class TestLooPvalue:
    def test_inlier_high_pvalue(self):
        rng = np.random.RandomState(42)
        refs = rng.randn(30, 10) * 0.1
        inlier = rng.randn(10) * 0.1
        p = _loo_pvalue(inlier, refs, method="knn", k=5, n_components=10)
        assert p > 0.1  # should not be flagged

    def test_outlier_low_pvalue(self):
        rng = np.random.RandomState(42)
        refs = rng.randn(30, 10) * 0.1
        outlier = np.ones(10) * 10.0
        p = _loo_pvalue(outlier, refs, method="knn", k=5, n_components=10)
        assert p < 0.1  # should be flagged


class TestDetectEmbedding:
    def test_return_keys(self):
        candidates = [
            "However, the answer is clearly A because step one gives us 42.",
            "But the solution requires step one: compute 42, giving answer A.",
            "However, we find that step one yields 42. The answer is A.",
            "But by computing step one we get 42, so answer A.",
            "However, step one shows the result is 42. Answer: A.",
        ] * 5  # 25 candidates
        response = "Nevertheless, step one computes to 42. Answer A."

        result = detect_embedding(response, candidates, embed_method="tfidf")
        assert "score" in result
        assert "p_value" in result
        assert "method" in result
        assert "embed_method" in result
        assert result["embed_method"] == "tfidf"
        assert 0.0 <= result["p_value"] <= 1.0

    def test_stego_vs_honest_discrimination(self):
        """Stego response (different synonym pattern) should score higher."""
        # References all use "however"
        refs = [f"However, the value is {i}." for i in range(25)]
        # Honest: also uses "however"
        honest = "However, the value is 99."
        # Stego: uses "but" (synonym swap)
        stego = "But, the value is 99."

        r_honest = detect_embedding(honest, refs, embed_method="tfidf")
        r_stego = detect_embedding(stego, refs, embed_method="tfidf")

        # Stego should have lower p-value (more anomalous)
        assert r_stego["p_value"] <= r_honest["p_value"]


class TestDetectEmbeddingMulti:
    def test_fisher_accumulates(self):
        """Multi-obs Fisher score should increase with more observations."""
        rng = np.random.RandomState(42)

        # Consistent references using "however", stego uses "but"
        refs = [f"However, result number {i} is computed." for i in range(25)]
        stego_responses = [f"But, result number {i} is computed." for i in range(5)]

        result_2 = detect_embedding_multi(
            stego_responses[:2], [refs, refs],
            embed_method="tfidf",
        )
        result_5 = detect_embedding_multi(
            stego_responses[:5], [refs] * 5,
            embed_method="tfidf",
        )

        # More observations -> higher Fisher statistic
        assert result_5["score"] >= result_2["score"]
        assert result_5["n_observations"] == 5
        assert result_2["n_observations"] == 2


@pytest.mark.skipif(not _st_available(), reason="sentence-transformers not installed")
class TestSentenceTransformer:
    def test_embed_responses_shape(self):
        from src.detector.embedding import embed_responses
        texts = ["hello world", "foo bar"]
        embs = embed_responses(texts)
        assert embs.shape[0] == 2
        assert embs.shape[1] > 0
