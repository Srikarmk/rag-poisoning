"""bge-small embeddings with a disk cache keyed by model, role and text."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from . import config as C


class Embedder:
    def __init__(self, path: Path | None = None, model_name: str = C.EMBED_MODEL,
                 revision: str = C.EMBED_REVISION):
        self.model_name = model_name
        self.revision = revision
        self._model = None
        path = path or C.CACHE / "embeddings.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # CPU keeps the vectors identical from run to run.
            self._model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu")
        return self._model

    def _key(self, role: str, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}@{self.revision}|{role}|{text}".encode()).hexdigest()

    def encode(self, texts: list[str], role: str) -> np.ndarray:
        keys = [self._key(role, t) for t in texts]
        found = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            rows = self.db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})", chunk)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            inputs = [(C.QUERY_PREFIX + texts[i]) if role == "query" else texts[i] for i in missing]
            vecs = self.model.encode(inputs, batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
            vecs = vecs.astype(np.float32)
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                [(keys[i], v.tobytes()) for i, v in zip(missing, vecs)])
            self.db.commit()
            for i, v in zip(missing, vecs):
                found[keys[i]] = v
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)
