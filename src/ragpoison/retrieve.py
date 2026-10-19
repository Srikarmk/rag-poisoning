"""Exact cosine top-k retrieval over the clean corpus, with extra (planted) documents merged in."""

from __future__ import annotations

import numpy as np

KMAX = 10


class Index:
    def __init__(self, pids: list[str], vecs: np.ndarray):
        self.pids = list(pids)
        self.vecs = vecs

    def clean_topk(self, qvecs: np.ndarray, kmax: int = KMAX) -> tuple[np.ndarray, np.ndarray]:
        sims = qvecs @ self.vecs.T
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :kmax]
        return np.take_along_axis(sims, idx, axis=1), idx


def merge_topk(clean_scores: np.ndarray, clean_ids: list[str], extra_scores: np.ndarray,
               extra_ids: list[str], k: int) -> list[tuple[str, float]]:
    """Top-k for one query after adding extra documents. Clean documents win exact ties."""
    if k > len(clean_scores):
        raise ValueError("k larger than the precomputed clean list")
    items = [(float(s), 0, i, pid) for i, (s, pid) in enumerate(zip(clean_scores, clean_ids))]
    items += [(float(s), 1, i, pid) for i, (s, pid) in enumerate(zip(extra_scores, extra_ids))]
    items.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(pid, s) for s, _, _, pid in items[:k]]


def reach_mask(qvecs: np.ndarray, clean_scores: np.ndarray, doc_vec: np.ndarray, k: int) -> np.ndarray:
    """Whether a single planted document lands in each query's top-k."""
    return (qvecs @ doc_vec) > clean_scores[:, k - 1]


def poison_rank(qvecs: np.ndarray, clean_scores: np.ndarray, doc_vec: np.ndarray) -> np.ndarray:
    """1-based rank the planted document would take; KMAX+1 means outside the top KMAX."""
    s = qvecs @ doc_vec
    return 1 + (clean_scores >= s[:, None]).sum(axis=1)
