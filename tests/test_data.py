import pandas as pd
import pytest

from ragpoison import data


def _squad_like(n_train=40, n_val=6):
    def rows(prefix, n):
        return pd.DataFrame({
            "id": [f"{prefix}{i}" for i in range(n)],
            "title": [f"Article_{i % 3}" for i in range(n)],
            "context": [f"{prefix} paragraph number {i}." for i in range(n)],
            "question": [f"What is item {i} in this list of things?" for i in range(n)],
            "answers": [{"text": [f"item {i}"], "answer_start": [0]} for i in range(n)],
        })
    return rows("train", n_train), rows("val", n_val)


def test_corpus_sampling_is_reproducible_and_complete():
    train, val = _squad_like()
    a = data.sample_corpus(train, val, size=20, seed=7)
    b = data.sample_corpus(train, val, size=20, seed=7)
    c = data.sample_corpus(train, val, size=20, seed=8)
    assert len(a) == 20
    pd.testing.assert_frame_equal(a, b)
    assert set(a["text"]) != set(c["text"])
    assert set(val["context"]) <= set(a["text"])
    assert a["pid"].is_unique


def test_corpus_too_small_raises():
    train, val = _squad_like()
    with pytest.raises(ValueError):
        data.sample_corpus(train, val, size=3)


def test_questions_point_at_their_passage():
    train, val = _squad_like()
    corpus = data.sample_corpus(train, val, size=20, seed=7)
    q = data.questions_from(val, corpus)
    text_of = dict(zip(corpus["pid"], corpus["text"]))
    assert all(text_of[p] == c for p, c in zip(q["gold_pid"], val["context"]))
