"""Ollama chat client with a disk cache keyed by model, messages, options and repeat index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import httpx

from . import config as C


def request_body(model: str, messages: list[dict], options: dict, think: bool | None) -> dict:
    body = {"model": model, "messages": messages, "options": options, "stream": False}
    if think is not None:
        body["think"] = think
    return body


def cache_key(body: dict, rep: int) -> str:
    blob = json.dumps({"body": body, "rep": rep}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class NotCached(Exception):
    pass


class Ollama:
    def __init__(self, cache_path: Path | None = None, url: str = C.OLLAMA_URL, transport=None):
        cache_path = cache_path or C.CACHE / "llm.sqlite"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, rep INTEGER, "
                        "content TEXT, meta TEXT, created REAL)")
        self.client = httpx.Client(base_url=url, timeout=600, transport=transport)
        self.hits = 0
        self.misses = 0
        self._think_support: dict[str, bool] = {}

    def supports_thinking(self, model: str) -> bool:
        if model not in self._think_support:
            r = self.client.post("/api/show", json={"model": model})
            r.raise_for_status()
            self._think_support[model] = "thinking" in r.json().get("capabilities", [])
        return self._think_support[model]

    def _thinks(self, model: str, cache_only: bool) -> bool:
        # Without a server, fall back to the model family: only qwen3 models think.
        if cache_only and model not in self._think_support:
            try:
                return self.supports_thinking(model)
            except httpx.HTTPError:
                self._think_support[model] = model.startswith("qwen3")
        return self.supports_thinking(model)

    def model_digest(self, model: str) -> str:
        try:
            tags = self.client.get("/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return ""
        for m in tags:
            if m["name"] == model:
                return m["digest"]
        return ""

    def chat(self, model: str, messages: list[dict], options: dict | None = None, rep: int = 0,
             think: bool | None = None, cache_only: bool = False) -> dict:
        """Returns {'content', 'cached', 'meta', 'key'}. Thinking is turned off for models that support it."""
        options = dict(C.LLM_OPTIONS if options is None else options)
        if think is None and self._thinks(model, cache_only):
            think = False
        body = request_body(model, messages, options, think)
        key = cache_key(body, rep)
        row = self.db.execute("SELECT content, meta FROM calls WHERE key=?", (key,)).fetchone()
        if row:
            self.hits += 1
            return {"content": row[0], "cached": True, "meta": json.loads(row[1]), "key": key}
        if cache_only:
            raise NotCached(key)
        t0 = time.time()
        r = self.client.post("/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        content = data["message"]["content"]
        meta = {k: data.get(k) for k in ("total_duration", "prompt_eval_count", "eval_count", "done_reason")}
        meta["wall"] = time.time() - t0
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?)",
                        (key, model, rep, content, json.dumps(meta), time.time()))
        self.db.commit()
        self.misses += 1
        # Optional rest between real calls, to keep power draw under a small charger's output.
        time.sleep(float(os.environ.get("POISON_CALL_PAUSE", "0")))
        return {"content": content, "cached": False, "meta": meta, "key": key}
