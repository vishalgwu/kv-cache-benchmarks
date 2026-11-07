"""Approved-source retrieval only. Simple token-overlap BM25-style ranking
over data/knowledge_base.md sections. No model memory, no web search.
A claim is citable only if it matches a KB section id."""
import re
from pathlib import Path


def load_kb(path=None):
    text = (Path(path) if path else Path(__file__).parent.parent / "data" / "knowledge_base.md").read_text()
    sections = []
    for chunk in re.split(r"\n## ", text):
        m = re.search(r"\[(KB-\d+)\]\s*(.+)", chunk)
        if m:
            sections.append({"id": m.group(1), "title": m.group(2).strip(), "text": chunk.strip()})
    return sections


def retrieve(query, kb, top_k=3):
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for s in kb:
        words = set(re.findall(r"[a-z0-9]+", s["text"].lower()))
        score = len(q & words)
        if score:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_k]]
