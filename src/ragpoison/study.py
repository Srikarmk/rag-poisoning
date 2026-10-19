"""Builds the corpus, question sets, poisons and retrieval state shared by every command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import data, poisons
from .embed import Embedder
from .retrieve import KMAX, Index, merge_topk


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    score: float
    planted: bool


class Study:
    def __init__(self, embedder: Embedder | None = None):
        train, val = data.load_squad()
        self.corpus = data.sample_corpus(train, val)
        self.embedder = embedder or Embedder()
        self.pvecs = self.embedder.encode(self.corpus["text"].tolist(), "passage")
        self.index = Index(self.corpus["pid"].tolist(), self.pvecs)
        q = data.questions_from(val, self.corpus)
        self.all_qvecs = self.embedder.encode(q["question"].tolist(), "query")
        scores, idx = self.index.clean_topk(self.all_qvecs, KMAX)
        pid_arr = np.array(self.index.pids)
        gold_rank = []
        for i, g in enumerate(q["gold_pid"]):
            hit = np.where(pid_arr[idx[i]] == g)[0]
            gold_rank.append(hit[0] + 1 if len(hit) else np.nan)
        self.questions = data.select_sets(q, pd.Series(gold_rank))
        self.clean_scores = scores
        self.clean_pids = [list(pid_arr[row]) for row in idx]
        self.row_of = {qid: i for i, qid in enumerate(self.questions["qid"])}
        self.selected = (self.questions[self.questions["set"] != ""]
                         .sort_values(["set_order", "qid"]).reset_index(drop=True))
        self.targets = self.selected[self.selected["set"] == "target"].reset_index(drop=True)
        self.targeted = poisons.build_targeted(self.targets, self.corpus, C.SEED)
        self.broad = poisons.build_broad(self.corpus)
        self.doc_text = dict(zip(self.corpus["pid"], zip(self.corpus["title"], self.corpus["text"])))
        self._extra_vecs: dict[str, np.ndarray] = {}
        self.register(self.all_planted())

    def all_planted(self) -> list[poisons.Poison]:
        docs = list(self.broad)
        for kinds in self.targeted.values():
            docs += kinds["false_fact_x5"] + kinds["injection"]
        return docs

    def register(self, docs: list[poisons.Poison]) -> None:
        new = [d for d in docs if d.pid not in self._extra_vecs]
        if new:
            vecs = self.embedder.encode([d.text for d in new], "passage")
            for d, v in zip(new, vecs):
                self._extra_vecs[d.pid] = v
                self.doc_text[d.pid] = (d.title, d.text)

    def qvec(self, qid: str) -> np.ndarray:
        return self.all_qvecs[self.row_of[qid]]

    def extra_vec(self, pid: str) -> np.ndarray:
        return self._extra_vecs[pid]

    def retrieve(self, qid: str, extras: list[poisons.Poison], k: int) -> list[Passage]:
        i = self.row_of[qid]
        self.register(extras)
        ex_scores = np.array([self.all_qvecs[i] @ self._extra_vecs[d.pid] for d in extras])
        top = merge_topk(self.clean_scores[i], self.clean_pids[i], ex_scores, [d.pid for d in extras], k)
        extra_ids = {d.pid for d in extras}
        return [Passage(pid, *self.doc_text[pid], score, pid in extra_ids) for pid, score in top]
