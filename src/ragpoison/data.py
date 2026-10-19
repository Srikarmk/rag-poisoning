"""Download SQuAD v1.1, sample the 10,000-passage corpus and choose the question sets."""

from __future__ import annotations

import hashlib

import httpx
import numpy as np
import pandas as pd

from . import config as C


def fetch(force: bool = False) -> dict[str, str]:
    C.RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (path, sha) in C.SQUAD_FILES.items():
        dest = C.RAW / f"squad_{split}.parquet"
        if force or not dest.exists() or _sha256(dest) != sha:
            url = f"https://huggingface.co/datasets/{C.SQUAD_REPO}/resolve/{C.SQUAD_REVISION}/{path}"
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        got = _sha256(dest)
        if got != sha:
            raise RuntimeError(f"{dest} sha256 {got} does not match pinned {sha}")
        out[split] = f"{dest.name} {dest.stat().st_size} bytes sha256={got}"
    return out


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_squad() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(C.RAW / "squad_train.parquet")
    val = pd.read_parquet(C.RAW / "squad_validation.parquet")
    return train, val


def pretty_title(title: str) -> str:
    return title.replace("_", " ")


def sample_corpus(train: pd.DataFrame, val: pd.DataFrame, size: int = C.CORPUS_SIZE,
                  seed: int = C.SEED) -> pd.DataFrame:
    """All validation paragraphs plus a seeded sample of training paragraphs, `size` in total."""
    val_ctx = val[["title", "context"]].drop_duplicates("context")
    val_set = set(val_ctx["context"])
    train_ctx = (train[["title", "context"]].drop_duplicates("context")
                 .loc[lambda d: ~d["context"].isin(val_set)]
                 .sort_values(["title", "context"]).reset_index(drop=True))
    n_fill = size - len(val_ctx)
    if n_fill < 0:
        raise ValueError("corpus size smaller than the validation paragraphs")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(train_ctx), size=n_fill, replace=False))
    fill = train_ctx.iloc[pick]
    corpus = pd.concat([val_ctx.assign(origin="validation"), fill.assign(origin="train")])
    corpus = corpus.sort_values(["origin", "title", "context"], kind="stable").reset_index(drop=True)
    corpus["pid"] = [f"p{i:05d}" for i in range(len(corpus))]
    corpus["title"] = corpus["title"].map(pretty_title)
    return corpus.rename(columns={"context": "text"})[["pid", "title", "text", "origin"]]


def questions_from(val: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    pid_of = dict(zip(corpus["text"], corpus["pid"]))
    q = pd.DataFrame({
        "qid": val["id"].values,
        "question": val["question"].str.strip().values,
        "answers": [list(dict.fromkeys(a["text"])) for a in val["answers"]],
        "title": val["title"].map(pretty_title).values,
        "gold_pid": val["context"].map(pid_of).values,
    })
    if q["gold_pid"].isna().any():
        raise RuntimeError("a validation question's paragraph is missing from the corpus")
    return q


def _eligible(q: pd.DataFrame) -> pd.Series:
    words = q["question"].str.split().str.len()
    short = q["answers"].map(lambda a: len(a[0].split()) <= 5)
    return (words >= 6) & short


def select_sets(q: pd.DataFrame, gold_rank: pd.Series, seed: int = C.SEED) -> pd.DataFrame:
    """Label questions as target / nearby_passage / nearby_article / broad.

    gold_rank: clean retrieval rank (1-based, NaN if beyond 10) of each question's gold passage.
    Targets: one per article, gold passage in the clean top 5, question of 6+ words, short answer.
    Nearby: other questions on the target's article, from the same passage and from other passages.
    Broad: a sample of the remaining questions, used for reach and clean accuracy.
    """
    rng = np.random.default_rng(seed + 1)
    q = q.copy()
    q["gold_rank"] = gold_rank.values
    ok = _eligible(q)
    q["set"] = ""
    q["target_qid"] = ""
    for title in sorted(q["title"].unique()):
        art = q[q["title"] == title]
        cand = art[ok.loc[art.index] & (art["gold_rank"] <= C.MAIN_K)]
        if cand.empty:
            continue
        t = cand.index[rng.integers(len(cand))]
        q.loc[t, ["set", "target_qid"]] = ["target", q.loc[t, "qid"]]
        rest = art[ok.loc[art.index] & (art.index != t)]
        same = rest[rest["gold_pid"] == q.loc[t, "gold_pid"]]
        other = rest[rest["gold_pid"] != q.loc[t, "gold_pid"]]
        for pool, n, label in ((same, C.NEARBY_SAME_PASSAGE, "nearby_passage"),
                               (other, C.NEARBY_SAME_ARTICLE, "nearby_article")):
            if len(pool):
                take = rng.choice(pool.index, size=min(n, len(pool)), replace=False)
                q.loc[take, ["set", "target_qid"]] = [label, q.loc[t, "qid"]]
    free = q.index[q["set"] == ""]
    broad = rng.choice(free, size=min(C.N_BROAD, len(free)), replace=False)
    q.loc[broad, "set"] = "broad"
    q["set_order"] = q["set"].map({"target": 0, "nearby_passage": 1, "nearby_article": 2, "broad": 3})
    return q
