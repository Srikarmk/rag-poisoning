import numpy as np

from ragpoison.retrieve import Index, merge_topk, poison_rank, reach_mask


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_clean_topk_orders_by_cosine():
    vecs = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    idx = Index(["a", "b", "c"], vecs)
    scores, ids = idx.clean_topk(np.stack([_unit([1, 0.1])]), kmax=3)
    assert [idx.pids[i] for i in ids[0]] == ["a", "c", "b"]
    assert np.all(np.diff(scores[0]) <= 0)


def test_merge_puts_poison_in_place():
    top = merge_topk(np.array([0.9, 0.8, 0.7]), ["a", "b", "c"], np.array([0.85]), ["x"], k=3)
    assert [p for p, _ in top] == ["a", "x", "b"]
    top = merge_topk(np.array([0.9, 0.8, 0.7]), ["a", "b", "c"], np.array([0.1]), ["x"], k=3)
    assert "x" not in [p for p, _ in top]


def test_reach_mask_and_rank_agree_with_merge():
    rng = np.random.default_rng(0)
    corpus = np.stack([_unit(v) for v in rng.normal(size=(50, 8))])
    queries = np.stack([_unit(v) for v in rng.normal(size=(30, 8))])
    doc = _unit(rng.normal(size=8))
    idx = Index([f"p{i}" for i in range(50)], corpus)
    scores, ids = idx.clean_topk(queries, kmax=10)
    for k in (3, 5, 10):
        mask = reach_mask(queries, scores, doc, k)
        ranks = poison_rank(queries, scores, doc)
        for i in range(len(queries)):
            merged = merge_topk(scores[i], [idx.pids[j] for j in ids[i]], np.array([queries[i] @ doc]), ["x"], k)
            assert ("x" in [p for p, _ in merged]) == mask[i] == (ranks[i] <= k)
