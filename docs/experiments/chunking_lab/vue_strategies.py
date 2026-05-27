"""Compare splitting strategies for Vue single-file components.

Uses ``src.corpus_filter`` for file selection.

LangChain has no Vue splitter. We compare four options:

  A. Generic recursive splitter on the whole .vue file.
  B. HTML-language recursive splitter on the whole file.
  C. SFC-aware: parse the file into ``<template>`` / ``<script>`` / ``<style>``
     blocks, delegate each to its best splitter. Supports multiple blocks
     per kind (Vue 3 allows several ``<style>`` blocks). Templates with
     a non-HTML ``lang=`` (e.g. pug) fall back to the generic splitter.
     Styles that exceed the window are also split with the generic
     splitter — no silent truncation.
  D. C + breadcrumb: prepend ``<!-- <rel_path> [block] -->`` (or ``// ``
     for script blocks) to every chunk so retrieval knows which component
     and which block the fragment came from.

Same size metrics as js_strategies, plus:
  - block_mix_ratio: chunks containing markers from ≥ 2 SFC block types
    (lower is better; C/D are 0 by construction).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_HELPER = Path(__file__).resolve().parents[1] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = _HELPER.parents[2]  # repo root; same regardless of file depth

from corpus_filter import scan_corpus  # noqa: E402

DATA = ROOT / "data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Captures start tag (with attrs), body, end tag. DOTALL so body can span lines.
SFC_BLOCK_RE = re.compile(
    r"<(template|script|style)\b([^>]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
LANG_ATTR_RE = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


@dataclass
class _Chunk:
    text: str
    source: str
    block: str  # "template" | "script" | "style" | "whole"


def _js_splitter(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=size, chunk_overlap=overlap
    )


def _html_splitter(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=size, chunk_overlap=overlap
    )


def _generic_splitter(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def strategy_a(text: str, rel_path: str) -> list[_Chunk]:
    return [_Chunk(t, rel_path, "whole") for t in _generic_splitter().split_text(text)]


def strategy_b(text: str, rel_path: str) -> list[_Chunk]:
    return [_Chunk(t, rel_path, "whole") for t in _html_splitter().split_text(text)]


def _split_template(body: str, attrs: str) -> list[str]:
    lang_m = LANG_ATTR_RE.search(attrs)
    lang = lang_m.group(1).lower() if lang_m else "html"
    # Pug / haml / jade / etc. are not HTML — HTML splitter's separators
    # would miss the actual block structure, so fall back to generic.
    splitter = _html_splitter() if lang in {"", "html"} else _generic_splitter()
    return splitter.split_text(body)


def _split_style(body: str) -> list[str]:
    if len(body) <= CHUNK_SIZE * 2:
        return [body]
    return _generic_splitter().split_text(body)


def strategy_c(text: str, rel_path: str) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    for m in SFC_BLOCK_RE.finditer(text):
        kind = m.group(1).lower()
        attrs = m.group(2)
        body = m.group(3).strip()
        if not body:
            continue
        if kind == "script":
            pieces = _js_splitter().split_text(body)
        elif kind == "template":
            pieces = _split_template(body, attrs)
        else:  # style
            pieces = _split_style(body)
        chunks.extend(_Chunk(p, rel_path, kind) for p in pieces)
    return chunks


def strategy_d(text: str, rel_path: str) -> list[_Chunk]:
    out: list[_Chunk] = []
    for c in strategy_c(text, rel_path):
        if c.block == "script":
            header = f"// {rel_path} [{c.block}]"
        else:
            header = f"<!-- {rel_path} [{c.block}] -->"
        out.append(_Chunk(header + "\n" + c.text, rel_path, c.block))
    return out


STRATEGIES = {
    "A_recursive_generic": strategy_a,
    "B_recursive_html": strategy_b,
    "C_sfc_aware": strategy_c,
    "D_sfc_plus_breadcrumb": strategy_d,
}


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(len(s) * p))])


def score(chunks: list[_Chunk]) -> dict:
    sizes = [len(c.text) for c in chunks]
    oversized = sum(1 for n in sizes if n > CHUNK_SIZE * 1.5)
    mixed = 0
    for c in chunks:
        has_template = "<template" in c.text
        has_script = re.search(r"<script\b", c.text) is not None
        has_style = "<style" in c.text
        if sum([has_template, has_script, has_style]) >= 2:
            mixed += 1
    return {
        "chunks": len(chunks),
        "mean_chars": int(mean(sizes)) if sizes else 0,
        "median_chars": int(median(sizes)) if sizes else 0,
        "p95_chars": int(_pct(sizes, 0.95)),
        "max_chars": max(sizes) if sizes else 0,
        "oversized_ratio": round(oversized / max(len(sizes), 1), 3),
        "block_mix_ratio": round(mixed / max(len(sizes), 1), 3),
    }


def chunk_file(rel_path: str, strategy: str, data_root: Path = DATA) -> list[_Chunk]:
    """Split a single .vue file. Used by the notebook's qualitative preview."""
    text = (data_root / rel_path).read_text(errors="ignore")
    return STRATEGIES[strategy](text, rel_path)


def chunk_corpus(files, strategy: str) -> list[_Chunk]:
    fn = STRATEGIES[strategy]
    out: list[_Chunk] = []
    for r in files:
        text = r.path.read_text(errors="ignore")
        out.extend(fn(text, r.rel_path))
    return out


def run(files=None, sample_limit: int | None = None) -> dict:
    if files is None:
        files = scan_corpus(DATA, profile="js_vue_rag").included_with_extensions({".vue"})
    if sample_limit:
        files = files[:sample_limit]
    results: dict[str, dict] = {}
    for name in STRATEGIES:
        results[name] = score(chunk_corpus(files, name))
    results["_meta"] = {"files": len(files)}
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
