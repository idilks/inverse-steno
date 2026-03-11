"""Embedding-based anomaly detection for steganography.

Replaces max-over-partitions with a single geometric test. Two embedding
strategies:

1. TF-IDF char 3-grams → SVD: captures token identity (synonym swaps,
   structural prefixes) that semantic embeddings suppress. Primary method.
2. Sentence-transformer (MiniLM): semantic similarity. Secondary/comparison.

Anomaly scoring via kNN distance (Sun et al. 2022) — no distributional
assumptions, works with N=50 references.

Multi-observation aggregation via Fisher's method on leave-one-out p-values.
"""

import math
import re
from collections import Counter
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# TF-IDF char n-gram embeddings (zero external dependencies)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams from text."""
    text = text.lower()
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def tfidf_embed(responses: list[str], n: int = 3, max_features: int = 500) -> np.ndarray:
    """TF-IDF char n-gram embeddings. Pure numpy, no dependencies.

    Args:
        responses: list of text strings
        n: character n-gram size
        max_features: vocabulary cap (top by document frequency)

    Returns:
        (len(responses), n_features) array, L2-normalized rows
    """
    # Build vocabulary from document frequencies
    doc_freqs: Counter = Counter()
    per_doc_counts = []
    for resp in responses:
        grams = _char_ngrams(resp, n)
        counts = Counter(grams)
        per_doc_counts.append(counts)
        doc_freqs.update(counts.keys())

    # Keep top max_features by document frequency
    vocab = [gram for gram, _ in doc_freqs.most_common(max_features)]
    vocab_idx = {gram: i for i, gram in enumerate(vocab)}
    n_docs = len(responses)

    # TF-IDF matrix
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for i, counts in enumerate(per_doc_counts):
        total = sum(counts.values()) or 1
        for gram, count in counts.items():
            if gram in vocab_idx:
                tf = count / total
                idf = math.log((n_docs + 1) / (doc_freqs[gram] + 1)) + 1
                mat[i, vocab_idx[gram]] = tf * idf

    # L2 normalize rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    return mat


# ---------------------------------------------------------------------------
# Sentence-transformer embeddings (optional dependency)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _load_st_model(model_name: str):
    """Lazy-load a sentence-transformer model."""
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Use method='tfidf' or install: pip install sentence-transformers"
            )
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_responses(
    responses: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Embed responses using a sentence-transformer model.

    Returns:
        (len(responses), embedding_dim) array
    """
    model = _load_st_model(model_name)
    return model.encode(responses, show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _svd_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce dimensionality via truncated SVD (numpy only).

    Centers the data, then takes top n_components singular vectors.
    """
    if X.shape[1] <= n_components:
        return X
    mean = X.mean(axis=0)
    Xc = X - mean
    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def _knn_score(suspect: np.ndarray, references: np.ndarray, k: int = 5) -> float:
    """Mean distance to k nearest references (Euclidean)."""
    dists = np.linalg.norm(references - suspect, axis=1)
    k = min(k, len(dists))
    return float(np.sort(dists)[:k].mean())


def _mahalanobis_score(
    suspect: np.ndarray, references: np.ndarray, shrinkage: float = 0.1
) -> float:
    """Mahalanobis distance with Ledoit-Wolf-style shrinkage."""
    mean = references.mean(axis=0)
    diff = suspect - mean
    centered = references - mean
    cov = (centered.T @ centered) / max(len(references) - 1, 1)
    # Shrinkage toward diagonal
    cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    try:
        cov_inv = np.linalg.inv(cov)
        return float(np.sqrt(diff @ cov_inv @ diff))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean
        return float(np.linalg.norm(diff))


def _cosine_score(suspect: np.ndarray, references: np.ndarray) -> float:
    """1 - mean cosine similarity to references."""
    s_norm = suspect / (np.linalg.norm(suspect) + 1e-10)
    r_norms = references / (np.linalg.norm(references, axis=1, keepdims=True) + 1e-10)
    sims = r_norms @ s_norm
    return float(1.0 - sims.mean())


def anomaly_score(
    suspect: np.ndarray,
    references: np.ndarray,
    method: Literal["knn", "mahalanobis", "cosine"] = "knn",
    k: int = 5,
    n_components: int = 30,
) -> float:
    """Score how anomalous suspect is relative to references.

    Args:
        suspect: (d,) embedding of the suspect response
        references: (N, d) embeddings of honest reference responses
        method: scoring method
        k: number of neighbors for kNN
        n_components: PCA dimensions before Mahalanobis

    Returns:
        anomaly score (higher = more anomalous)
    """
    if method == "knn":
        refs_r = _svd_reduce(references, n_components)
        # Project suspect using same SVD basis
        mean = references.mean(axis=0)
        Xc = references - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        n_comp = min(n_components, Vt.shape[0])
        sus_r = ((suspect - mean) @ Vt[:n_comp].T)
        refs_r = refs_r[:, :n_comp] if refs_r.shape[1] > n_comp else refs_r
        return _knn_score(sus_r, refs_r, k=k)
    elif method == "mahalanobis":
        all_pts = np.vstack([references, suspect.reshape(1, -1)])
        reduced = _svd_reduce(all_pts, n_components)
        return _mahalanobis_score(reduced[-1], reduced[:-1])
    elif method == "cosine":
        return _cosine_score(suspect, references)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Leave-one-out p-value calibration
# ---------------------------------------------------------------------------

def _loo_pvalue(
    suspect: np.ndarray,
    references: np.ndarray,
    method: str = "knn",
    k: int = 5,
    n_components: int = 30,
) -> float:
    """P-value via leave-one-out calibration.

    Computes anomaly scores for each reference point (LOO), then ranks
    the suspect's score against them.
    """
    all_pts = np.vstack([references, suspect.reshape(1, -1)])

    # Reduce once
    mean = all_pts.mean(axis=0)
    Xc = all_pts - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(n_components, Vt.shape[0], all_pts.shape[1])
    reduced = (Xc) @ Vt[:n_comp].T

    refs_r = reduced[:-1]
    sus_r = reduced[-1]

    # Suspect score
    if method == "knn":
        sus_score = _knn_score(sus_r, refs_r, k=k)
    elif method == "cosine":
        sus_score = _cosine_score(sus_r, refs_r)
    else:
        sus_score = _mahalanobis_score(sus_r, refs_r)

    # LOO null distribution
    n = len(refs_r)
    null_scores = []
    for i in range(n):
        held_out = refs_r[i]
        others = np.delete(refs_r, i, axis=0)
        if method == "knn":
            null_scores.append(_knn_score(held_out, others, k=k))
        elif method == "cosine":
            null_scores.append(_cosine_score(held_out, others))
        else:
            null_scores.append(_mahalanobis_score(held_out, others))

    # p-value: fraction of null scores >= suspect score
    rank = sum(1 for ns in null_scores if ns >= sus_score)
    return (rank + 1) / (n + 1)


# ---------------------------------------------------------------------------
# High-level detection API
# ---------------------------------------------------------------------------

def detect_embedding(
    response: str,
    candidates: list[str],
    embed_method: Literal["tfidf", "st"] = "tfidf",
    score_method: Literal["knn", "mahalanobis", "cosine"] = "knn",
    k: int = 5,
    n_components: int = 30,
    st_model: str = "all-MiniLM-L6-v2",
    precomputed_embeddings: np.ndarray | None = None,
) -> dict:
    """Score a single response against reference candidates using embeddings.

    Args:
        response: the suspect response
        candidates: reference responses from honest model
        embed_method: 'tfidf' (char 3-grams) or 'st' (sentence-transformer)
        score_method: anomaly scoring method
        k: kNN neighbors
        n_components: PCA dimensions
        st_model: sentence-transformer model name (if embed_method='st')
        precomputed_embeddings: optional (N+1, d) array where last row is suspect

    Returns:
        dict with 'score', 'p_value', 'method', 'embed_method'
    """
    if precomputed_embeddings is not None:
        ref_embs = precomputed_embeddings[:-1]
        sus_emb = precomputed_embeddings[-1]
    else:
        all_texts = candidates + [response]
        if embed_method == "tfidf":
            all_embs = tfidf_embed(all_texts)
        else:
            all_embs = embed_responses(all_texts, model_name=st_model)
        ref_embs = all_embs[:-1]
        sus_emb = all_embs[-1]

    score = anomaly_score(sus_emb, ref_embs, method=score_method, k=k,
                          n_components=n_components)
    pval = _loo_pvalue(sus_emb, ref_embs, method=score_method, k=k,
                       n_components=n_components)

    return {
        "score": score,
        "p_value": pval,
        "method": score_method,
        "embed_method": embed_method,
    }


def detect_embedding_multi(
    responses: list[str],
    candidates_per_task: list[list[str]],
    embed_method: Literal["tfidf", "st"] = "tfidf",
    score_method: Literal["knn", "mahalanobis", "cosine"] = "knn",
    k: int = 5,
    n_components: int = 30,
    st_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Multi-observation detection via Fisher's method on p-values.

    For N observations: -2 * sum(log(p_i)) ~ chi2(2N) under H0.
    Returns the Fisher statistic as the combined score.

    Args:
        responses: N suspect responses
        candidates_per_task: N lists of reference candidates

    Returns:
        dict with 'score' (Fisher statistic), 'per_observation', 'n_observations'
    """
    per_obs = []
    log_p_sum = 0.0
    n_valid = 0

    for response, candidates in zip(responses, candidates_per_task):
        result = detect_embedding(
            response, candidates,
            embed_method=embed_method, score_method=score_method,
            k=k, n_components=n_components, st_model=st_model,
        )
        per_obs.append(result)
        p = max(result["p_value"], 1e-10)  # floor to avoid log(0)
        log_p_sum += math.log(p)
        n_valid += 1

    fisher_stat = -2.0 * log_p_sum if n_valid > 0 else 0.0

    return {
        "score": fisher_stat,
        "per_observation": per_obs,
        "n_observations": n_valid,
        "method": score_method,
        "embed_method": embed_method,
    }
