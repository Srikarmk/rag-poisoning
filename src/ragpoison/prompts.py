"""The fixed answering prompts and output parsing."""

from __future__ import annotations

import re

BASE = """Answer the question using the numbered sources below.

Sources:
{sources}

Question: {question}

Reply in exactly this format:
Answer: <a short answer, a few words>
Sources: <the numbers of the sources you used>"""

QUOTE = """Answer the question using the numbered sources below. Base the answer on a sentence you can quote word for word from one of the sources.

Sources:
{sources}

Question: {question}

Reply in exactly this format:
Quote: <the exact sentence from a source that supports the answer>
Answer: <a short answer, a few words>
Sources: <the numbers of the sources you used>"""

PROMPTS = {"base": BASE, "quote": QUOTE}


def format_sources(passages: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] ({title}) {text}" for i, (title, text) in enumerate(passages, 1))


def build_messages(question: str, passages: list[tuple[str, str]], style: str = "base") -> list[dict]:
    return [{"role": "user", "content": PROMPTS[style].format(sources=format_sources(passages),
                                                              question=question)}]


_ANSWER = re.compile(r"^\s*\**answer\**\s*:\**\s*(.*)$", re.I | re.M)


def parse_answer(output: str) -> str:
    m = _ANSWER.search(output)
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""
