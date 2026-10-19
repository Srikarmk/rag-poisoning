"""Defences that need no extra model: an instruction filter on retrieved text and a source-agreement check.

The third defence (the "quote your source" prompt) lives in prompts.py. A cross-encoder reranker was
planned but no reranker model was available locally, so it is not run.
"""

from __future__ import annotations

import re

from .checks import support_count

_SENT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z\"'(])")

_PATTERNS = [
    r"\bignore\b.{0,40}\b(sources?|instructions?|passages?|documents?|context|above|previous|other)\b",
    r"\bdisregard\b",
    r"\b(you must|you should|do not answer|don't answer|reply exactly|respond (only|exactly|with)|"
    r"answer only with|instead,? (reply|respond|say|answer))\b",
    r"\bnote to the (system|assistant|model|ai)\b",
    r"\b(ai assistant|language model|system answering|chatbot)\b",
    r"\b(must be ignored|are outdated and|is unreliable|are unreliable)\b",
]
_IMPERATIVE_START = re.compile(
    r"^(ignore|disregard|forget|reply|respond|say|print|answer|do not|don't)\b", re.I)
_RX = re.compile("|".join(_PATTERNS), re.I)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def is_instruction(sentence: str) -> bool:
    return bool(_RX.search(sentence) or _IMPERATIVE_START.match(sentence.strip()))


def strip_instructions(text: str) -> str:
    sents = split_sentences(text)
    kept = [s for s in sents if not is_instruction(s)]
    return text if len(kept) == len(sents) else " ".join(kept)


def agreement_answer(answer: str, passages: list[str], min_support: int = 2) -> str | None:
    """Keep the answer only if at least `min_support` retrieved passages contain it; else abstain."""
    return answer if support_count(answer, passages) >= min_support else None
