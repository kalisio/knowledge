"""JavaScript chunking with path and symbol breadcrumbs.

``RecursiveCharacterTextSplitter.from_language(Language.JS)`` drives the
cut points, then a ``// <rel_path> :: <nearest symbol>`` header is
prepended to every chunk so the embedding model has a stable path and
symbol anchor in the text itself.
"""

from __future__ import annotations

import re

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

JS_CHUNK_SIZE = 800
JS_CHUNK_OVERLAP = 120

_TOP_LEVEL_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


def _nearest_js_symbol(text: str, offset: int) -> str:
    last = ""
    for m in _TOP_LEVEL_SYMBOL_RE.finditer(text):
        if m.start() > offset:
            break
        last = m.group(1)
    return last


def chunk_js(
    text: str,
    source: str,
    *,
    chunk_size: int = JS_CHUNK_SIZE,
    chunk_overlap: int = JS_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk one JavaScript file with path and symbol breadcrumbs."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    pieces = splitter.split_text(text)
    out: list[dict] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        pos = text.find(piece[:32], cursor) if piece else -1
        if pos < 0:
            pos = cursor
        symbol = _nearest_js_symbol(text, pos)
        header = f"// {source}" + (f" :: {symbol}" if symbol else "")
        breadcrumb = {"path": source, "symbol": symbol, "block": ""}
        out.append({
            "text": header + "\n" + piece,
            "metadata": {
                "source": source,
                "strategy": "D_js_breadcrumb",
                "chunk_index": i,
                "breadcrumb": breadcrumb,
            },
        })
        cursor = pos + len(piece)
    return out


JS_WINNER = "D_js_breadcrumb"
