"""Compare chunking strategies on the JSON slice of the corpus.

Named ``json_splitter_experiment.py`` to mirror nb03's
``js_splitter_experiment.py`` / ``vue_splitter_experiment.py`` layout —
one experiment module per file type, driven by the notebook.

Four strategies per JSON category:

    A. RCT baseline — generic recursive character splitter; treats the
       file as plain text. Blind to JSON structure.
    B. RecursiveJsonSplitter — LangChain's JSON-aware splitter, descends
       the tree until each sub-tree fits. Every chunk is itself a valid
       JSON subtree.
    C. Category-aware key split — delegates to ``chunk_json`` from the
       production package, then strips the breadcrumb header so C's
       chunks are exactly the splitter's output (for a fair comparison
       against A/B, which emit no header).
    D. C + breadcrumb — same as C but keeps the ``// <rel_path> :: <unit>``
       header that the production chunker prepends. Comments are not
       legal JSON, so after D the chunk text is not a parseable JSON
       document — this is intentional (the chunk is for the embedding
       model, not for a JSON consumer).

Metrics:
    chunks          — total count (indexing cost)
    mean / median / p95 / max char size
    oversized_ratio — fraction exceeding target_size * 1.5 (runaway)
    parse_integrity — fraction of chunks that parse as standalone JSON
                      after stripping any ``//`` header line. Only
                      meaningful for B / C / D where chunks should be
                      structurally complete.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "nb04_chunking_json"))

from corpus_filter import scan_corpus  # noqa: E402
from chunking import chunk_json  # noqa: E402  — production category-aware splitter
from chunking.json_chunking import JSON_CHUNK_SIZE  # noqa: E402
from json_inventory import JsonFileView, categorize, load_views  # noqa: E402

DATA = ROOT / "data"

CHUNK_SIZE = JSON_CHUNK_SIZE
CHUNK_OVERLAP = 80

_HEADER_LINE_RE = re.compile(r"^//[^\n]*\n")


@dataclass
class JsonChunk:
    text: str
    source: str
    unit: str
    category: str


# ── A. RCT baseline ────────────────────────────────────────

def _rct_split(view: JsonFileView) -> list[JsonChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pieces = splitter.split_text(view.text)
    return [
        JsonChunk(text=p, source=view.rel_path, unit="<rct>", category=view.category)
        for p in pieces
    ]


# ── B. RecursiveJsonSplitter ───────────────────────────────

def _rjs_split(view: JsonFileView) -> list[JsonChunk]:
    if not view.parse_ok:
        return _rct_split(view)
    splitter = RecursiveJsonSplitter(
        max_chunk_size=CHUNK_SIZE, min_chunk_size=max(50, CHUNK_SIZE // 4)
    )
    subtrees = splitter.split_json(json_data=view.data, convert_lists=True)
    return [
        JsonChunk(
            text=json.dumps(st, ensure_ascii=False),
            source=view.rel_path, unit="<subtree>", category=view.category,
        )
        for st in subtrees
    ]


# ── C/D. Delegate to the production chunker ────────────────
# ``chunk_json`` always emits a breadcrumb header; C strips it so A/B/C
# are compared on the splitter output only, D keeps it.

def _production_split(view: JsonFileView, *, with_breadcrumb: bool) -> list[JsonChunk]:
    out: list[JsonChunk] = []
    for c in chunk_json(view.text, view.rel_path):
        text = c["text"]
        if not with_breadcrumb:
            text = _HEADER_LINE_RE.sub("", text, count=1)
        unit = c["metadata"]["breadcrumb"]["symbol"] or c["metadata"]["breadcrumb"]["block"]
        out.append(JsonChunk(
            text=text, source=view.rel_path,
            unit=unit, category=view.category,
        ))
    return out


def _category_split(view: JsonFileView) -> list[JsonChunk]:
    return _production_split(view, with_breadcrumb=False)


def _category_breadcrumb_split(view: JsonFileView) -> list[JsonChunk]:
    return _production_split(view, with_breadcrumb=True)


# ── strategy registry ─────────────────────────────────────

STRATEGIES: dict[str, Callable[[JsonFileView], list[JsonChunk]]] = {
    "A_rct":                      _rct_split,
    "B_recursive_json":           _rjs_split,
    "C_category_aware":           _category_split,
    "D_category_plus_breadcrumb": _category_breadcrumb_split,
}


# ── metrics ────────────────────────────────────────────────

def _pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return int(s[min(len(s) - 1, int(len(s) * p))])


def _parse_integrity(chunks: list[JsonChunk], strip_header: bool) -> float:
    if not chunks:
        return 0.0
    ok = 0
    for c in chunks:
        text = _HEADER_LINE_RE.sub("", c.text, count=1) if strip_header else c.text
        try:
            json.loads(text)
            ok += 1
        except Exception:
            continue
    return round(ok / len(chunks), 3)


def score_chunks(chunks: list[JsonChunk], target_size: int, *, strip_header: bool = False) -> dict:
    sizes = [len(c.text) for c in chunks]
    if not sizes:
        return {"chunks": 0}
    oversized = sum(1 for n in sizes if n > target_size * 1.5)
    return {
        "chunks": len(chunks),
        "mean_chars": int(mean(sizes)),
        "median_chars": int(median(sizes)),
        "p95_chars": _pct(sizes, 0.95),
        "max_chars": max(sizes),
        "oversized_ratio": round(oversized / len(sizes), 3),
        "parse_integrity": _parse_integrity(chunks, strip_header=strip_header),
    }


# ── public entry ────────────────────────────────────────

def chunk_corpus(views: list[JsonFileView], strategy: str) -> list[JsonChunk]:
    splitter = STRATEGIES[strategy]
    out: list[JsonChunk] = []
    for v in views:
        out.extend(splitter(v))
    return out


def chunk_file(rel_path: str, strategy: str, data_root: Path = DATA) -> list[JsonChunk]:
    """Split one file by rel_path. Used by the notebook's qualitative preview."""
    p = data_root / rel_path
    text = p.read_text(errors="ignore")
    try:
        data = json.loads(text)
        parse_ok = True
    except Exception:
        data = None
        parse_ok = False
    view = JsonFileView(
        rel_path=rel_path, size=p.stat().st_size, text=text,
        category=categorize(rel_path), data=data, parse_ok=parse_ok,
    )
    return STRATEGIES[strategy](view)


def run(views: list[JsonFileView] | None = None, categories: set[str] | None = None) -> dict:
    """Run every strategy and return per-strategy + per-category metrics.

    If ``categories`` is given, restrict the corpus to those inventory
    categories (e.g. ``{"schemas_validation", "i18n_translations"}`` for
    a RAG-focused view).
    """
    if views is None:
        recs = scan_corpus(DATA, profile="js_vue_rag").included_with_extensions({".json"})
        views = load_views(recs)
    if categories is not None:
        views = [v for v in views if v.category in categories]

    overall: dict[str, dict] = {}
    by_cat: dict[str, dict[str, dict]] = {}

    for strat in STRATEGIES:
        chunks = chunk_corpus(views, strat)
        overall[strat] = score_chunks(
            chunks, CHUNK_SIZE,
            strip_header=strat.startswith("D_"),
        )
        by_cat_chunks: dict[str, list[JsonChunk]] = {}
        for c in chunks:
            by_cat_chunks.setdefault(c.category, []).append(c)
        for cat, cchunks in by_cat_chunks.items():
            by_cat.setdefault(cat, {})[strat] = score_chunks(
                cchunks, CHUNK_SIZE, strip_header=strat.startswith("D_"),
            )

    return {
        "_meta": {"files": len(views), "chunk_size": CHUNK_SIZE},
        "overall": overall,
        "by_category": by_cat,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
