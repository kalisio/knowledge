"""Public dispatcher for the chunking package.

``chunk_files()`` walks a batch of ``corpus_filter`` records and
dispatches each by file extension to the per-type winner chunker
(``chunk_markdown`` / ``chunk_js`` / ``chunk_vue`` / ``chunk_json``).
"""

from __future__ import annotations

from typing import Iterable

from ._line_range import compute_line_range
from .js import chunk_js
from .json_chunking import JSON_INDEXED_CATEGORIES, chunk_json, json_category
from .markdown import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, MD_WINNER, chunk_markdown
from .vue import chunk_vue


_EXT_DISPATCH = {
    ".md": "markdown",
    ".js": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".vue": "vue",
    ".json": "json",
}


def chunk_files(
    files: Iterable,
    *,
    md_strategy: str = MD_WINNER,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    json_categories: set[str] | None = JSON_INDEXED_CATEGORIES,
) -> list[dict]:
    """Chunk a batch of corpus_filter records, dispatching by extension.

    Accepts any iterable of objects exposing ``.path`` and ``.rel_path``.

    ``json_categories`` filters JSON files by semantic role — defaults
    to the set nb04 recommends for the production index. Pass ``None``
    to include every JSON regardless of category.
    """
    out: list[dict] = []
    for rec in files:
        text = rec.path.read_text(encoding="utf-8", errors="ignore")
        src = str(rec.rel_path)
        ext = rec.path.suffix.lower()
        kind = _EXT_DISPATCH.get(ext)
        if kind == "markdown":
            chunks = chunk_markdown(
                text, src, strategy=md_strategy,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
        elif kind == "js":
            chunks = chunk_js(text, src)
        elif kind == "vue":
            chunks = chunk_vue(text, src)
        elif kind == "json":
            if json_categories is not None and json_category(src) not in json_categories:
                continue
            chunks = chunk_json(text, src)
        else:
            continue

        for chunk in chunks:
            start_line, end_line = compute_line_range(chunk["text"], text)
            metadata = chunk.setdefault("metadata", {})
            metadata["start_line"] = start_line
            metadata["end_line"] = end_line
        out.extend(chunks)
    return out
