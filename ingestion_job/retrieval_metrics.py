"""Deterministic retrieval metrics for source-level evaluation.

All metrics are reproducible from gold source files and/or target symbols —
no LLM judge is needed. They are aligned with how an AI coder agent actually
consumes retrieval output (Top-K bundle, not individual ranked chunks).
"""

from __future__ import annotations

import re


def recall_at_k(
    retrieved_sources: list[str],
    gold_sources: list[str],
    k: int | None = None,
) -> float:
    """Fraction of gold source files present in the Top-K retrieved sources.

    A value of 1.0 means every gold source file was retrieved at least once
    in the Top-K; 0.0 means none of them were.
    """
    if not gold_sources:
        return 0.0
    top = retrieved_sources[:k] if k is not None else retrieved_sources
    top_set = set(top)
    hits = sum(1 for g in gold_sources if g in top_set)
    return hits / len(gold_sources)


def hit_at_k(
    retrieved_sources: list[str],
    gold_sources: list[str],
    k: int | None = None,
) -> int:
    """1 if at least one gold source appears in Top-K, else 0."""
    if not gold_sources:
        return 0
    top = retrieved_sources[:k] if k is not None else retrieved_sources
    return int(any(g in top for g in gold_sources))


def symbol_hit(text: str, symbols: list[str]) -> bool:
    """True if every symbol appears (word-boundary match) in text.

    Symbols are API identifiers (function names, class names, CLI flags).
    We use word-boundary matching to avoid e.g. matching ``Map`` inside ``Mapping``.
    """
    if not symbols:
        return False
    for sym in symbols:
        pattern = rf"\b{re.escape(sym)}\b"
        if not re.search(pattern, text):
            return False
    return True


def symbol_hit_rate(bundle_text: str, symbols: list[str]) -> float:
    """Fraction of target symbols present in the bundle."""
    if not symbols:
        return 0.0
    found = sum(
        1 for sym in symbols if re.search(rf"\b{re.escape(sym)}\b", bundle_text)
    )
    return found / len(symbols)


def code_integrity_rate(chunk_texts: list[str]) -> float:
    """Fraction of chunks with *balanced* code fences (```).

    A value of 1.0 means no chunk cut a code block in half. A value of 0.85
    means 15% of chunks have an orphan fence and an agent reading that chunk
    would see a truncated code example.
    """
    if not chunk_texts:
        return 1.0
    ok = sum(1 for t in chunk_texts if t.count("```") % 2 == 0)
    return ok / len(chunk_texts)


def bundle_token_efficiency(bundle_text: str, gold_symbols: list[str]) -> float:
    """Rough estimate: `relevant characters / total characters` in the bundle.

    We approximate "relevant" by counting characters inside any line that
    mentions a gold symbol. It's crude, but gives a reproducible number to
    compare strategies on how much of the Top-K bundle is actually on-topic
    versus filler. A higher value means the agent wastes less context window.
    """
    if not bundle_text or not gold_symbols:
        return 0.0
    patterns = [re.compile(rf"\b{re.escape(s)}\b") for s in gold_symbols]
    relevant = 0
    for line in bundle_text.splitlines(keepends=True):
        if any(p.search(line) for p in patterns):
            relevant += len(line)
    return relevant / len(bundle_text)
