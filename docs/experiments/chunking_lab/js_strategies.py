"""Compare LangChain text splitters on the JS corpus.

Uses ``src.corpus_filter`` for file selection — minified bundles, oversized
data files and other noise are excluded before any splitting happens.

Strategies compared:
  A. Recursive (generic separators) — baseline, language-unaware.
  B. Recursive from_language(JS) — uses JS-aware separators.
  C. Recursive from_language(JS) with a larger window tuned for methods.
  D. B + breadcrumb: prepend ``// <rel_path> :: <nearest top-level symbol>``
     to every chunk.
  D_path_only. B + path-only breadcrumb (``// <rel_path>`` without
     the symbol) — disambiguation variant to measure how much of D's
     retrieval lift comes from token overlap with the symbol name.

Metrics per strategy:
  - chunk count
  - char size distribution (mean / median / p95 / max)
  - oversized_ratio: fraction of chunks exceeding target_size * 1.5
  - boundary_quality: fraction of chunks that *start* at a structurally
    clean JS boundary.  For breadcrumb strategies the header line is
    stripped before evaluation so the score reflects the splitter alone.

Boundary quality is a rough proxy for "would a retriever pull a coherent
unit?"; it is NOT a retrieval metric. For that see ``retrieval_eval.py``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_HELPER = Path(__file__).resolve().parents[1] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_filter import scan_corpus  # noqa: E402

DATA = ROOT / "data"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Clean-start heuristic. Deliberately excludes a bare `*` line because
# those are usually the middle of a JSDoc block, not a fresh unit.
CLEAN_START_RE = re.compile(
    r"^\s*(?:"
    r"import\b|export\b|function\b|class\b|async\s+function\b|"
    r"const\b|let\b|var\b|//|/\*|"
    r"[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{"   # method signature
    r")"
)

TOP_LEVEL_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


@dataclass
class _Chunk:
    text: str
    source: str  # rel_path


def _recursive_generic() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def _recursive_js(size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=size, chunk_overlap=overlap
    )


STRATEGIES: dict[str, tuple[Callable[[], RecursiveCharacterTextSplitter], int]] = {
    "A_recursive_generic": (_recursive_generic, CHUNK_SIZE),
    "B_recursive_js": (_recursive_js, CHUNK_SIZE),
    "C_recursive_js_large": (lambda: _recursive_js(1400, 200), 1400),
    "D_js_plus_breadcrumb": (_recursive_js, CHUNK_SIZE),
    "D_path_only": (_recursive_js, CHUNK_SIZE),
}


def _nearest_symbol(text: str, offset: int) -> str:
    """Name of the last top-level symbol declared at or before *offset*."""
    last = ""
    for m in TOP_LEVEL_SYMBOL_RE.finditer(text):
        if m.start() > offset:
            break
        last = m.group(1)
    return last


def _split_with_breadcrumb(file_text: str, rel_path: str, *, include_symbol: bool = True) -> list[str]:
    splitter = _recursive_js()
    pieces = splitter.split_text(file_text)
    out: list[str] = []
    cursor = 0
    for piece in pieces:
        pos = file_text.find(piece[:32], cursor) if piece else -1
        if pos < 0:
            pos = cursor
        if include_symbol:
            symbol = _nearest_symbol(file_text, pos)
            header = f"// {rel_path}" + (f" :: {symbol}" if symbol else "")
        else:
            header = f"// {rel_path}"
        out.append(header + "\n" + piece)
        cursor = pos + len(piece)
    return out


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(len(s) - 1, int(len(s) * p))])


def score_chunks(chunks: list[str], target_size: int, *, strip_first_line: bool = False) -> dict:
    sizes = [len(c) for c in chunks]
    oversized = sum(1 for n in sizes if n > target_size * 1.5)
    bq_texts = [c.partition("\n")[2] for c in chunks] if strip_first_line else chunks
    clean = sum(1 for c in bq_texts if CLEAN_START_RE.match(c))
    return {
        "chunks": len(chunks),
        "mean_chars": int(mean(sizes)) if sizes else 0,
        "median_chars": int(median(sizes)) if sizes else 0,
        "p95_chars": int(_pct(sizes, 0.95)),
        "max_chars": max(sizes) if sizes else 0,
        "oversized_ratio": round(oversized / max(len(sizes), 1), 3),
        "boundary_quality": round(clean / max(len(sizes), 1), 3),
    }


def chunk_file(rel_path: str, strategy: str, data_root: Path = DATA) -> list[_Chunk]:
    """Split a single .js file. Used by the notebook's qualitative preview."""
    text = (data_root / rel_path).read_text(errors="ignore")
    if strategy in _BREADCRUMB_STRATEGIES:
        include_sym = strategy == "D_js_plus_breadcrumb"
        return [_Chunk(text=p, source=rel_path)
                for p in _split_with_breadcrumb(text, rel_path, include_symbol=include_sym)]
    factory, _ = STRATEGIES[strategy]
    return [_Chunk(text=p, source=rel_path) for p in factory().split_text(text)]


def chunk_corpus(files, strategy: str) -> list[_Chunk]:
    """Split every file under *files* with the named strategy. Returns
    (text, source) tuples so retrieval eval can group chunks by file."""
    out: list[_Chunk] = []
    if strategy in ("D_js_plus_breadcrumb", "D_path_only"):
        include_sym = strategy == "D_js_plus_breadcrumb"
        for r in files:
            text = r.path.read_text(errors="ignore")
            for piece in _split_with_breadcrumb(text, r.rel_path, include_symbol=include_sym):
                out.append(_Chunk(text=piece, source=r.rel_path))
        return out

    factory, _ = STRATEGIES[strategy]
    splitter = factory()
    for r in files:
        text = r.path.read_text(errors="ignore")
        for piece in splitter.split_text(text):
            out.append(_Chunk(text=piece, source=r.rel_path))
    return out


_BREADCRUMB_STRATEGIES = {"D_js_plus_breadcrumb", "D_path_only"}


def run(files=None, sample_limit: int | None = None) -> dict:
    if files is None:
        files = scan_corpus(DATA, profile="js_vue_rag").included_with_extensions({".js"})
    if sample_limit:
        files = files[:sample_limit]
    results: dict[str, dict] = {}
    total_chars = 0
    for name, (_, target) in STRATEGIES.items():
        chunks = chunk_corpus(files, name)
        if name == "A_recursive_generic":
            total_chars = sum(len(c.text) for c in chunks)
        strip = name in _BREADCRUMB_STRATEGIES
        results[name] = score_chunks([c.text for c in chunks], target, strip_first_line=strip)
    results["_meta"] = {"files": len(files), "total_chars_strategy_A": total_chars}
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
