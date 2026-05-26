"""Vue SFC chunking with block-aware breadcrumbs.

Dispatches on SFC block (``<template>`` / ``<script>`` / ``<style>``),
splits each with the appropriate language-aware splitter, then prepends
a breadcrumb header that includes a paraphrased component name so the
embedding model sees surface tokens shared with natural-language
queries (``"zoom control"`` matches ``KZoomControl``).

``strategy="D_vue_breadcrumb"`` selects the path-only variant (no
paraphrased phrase) for callers that prefer minimal per-chunk overhead.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .js import JS_CHUNK_OVERLAP, JS_CHUNK_SIZE

_SFC_BLOCK_RE = re.compile(
    r"<(template|script|style)\b([^>]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_LANG_ATTR_RE = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_VUE_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _component_phrase(source: str) -> str:
    """``KZoomControl.vue`` → ``"zoom control"``. Consecutive capitals
    stay as one acronym token (``KHTTPClient`` → ``"http client"``);
    the leading ``K`` prefix and 1-char shards are dropped."""
    stem = Path(source).stem
    name = stem[1:] if stem.startswith("K") and len(stem) > 1 else stem
    words = [w.lower() for w in _VUE_CAMEL_RE.split(name) if w]
    words = [w for w in words if len(w) > 1]
    return " ".join(words)


def _split_template(body: str, attrs: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    lang_m = _LANG_ATTR_RE.search(attrs)
    lang = lang_m.group(1).lower() if lang_m else "html"
    if lang in {"", "html"}:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return splitter.split_text(body)


def _split_style(body: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(body) <= chunk_size * 2:
        return [body]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_text(body)


def chunk_vue(
    text: str,
    source: str,
    *,
    strategy: str = "E_vue_expanded_breadcrumb",
    chunk_size: int = JS_CHUNK_SIZE,
    chunk_overlap: int = JS_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk one Vue SFC.

    - ``E_vue_expanded_breadcrumb`` (default): split SFC blocks and add
      a breadcrumb that includes the paraphrased component name.
    - ``D_vue_breadcrumb``: path-only breadcrumb.
    """
    if strategy not in ("E_vue_expanded_breadcrumb", "D_vue_breadcrumb"):
        raise ValueError(
            f"Unknown Vue strategy {strategy!r}. Choose E_vue_expanded_breadcrumb "
            "or D_vue_breadcrumb."
        )

    phrase = _component_phrase(source) if strategy == "E_vue_expanded_breadcrumb" else ""

    out: list[dict] = []
    idx = 0
    for m in _SFC_BLOCK_RE.finditer(text):
        kind = m.group(1).lower()
        attrs = m.group(2)
        body = m.group(3).strip()
        if not body:
            continue

        if kind == "script":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            pieces = splitter.split_text(body)
        elif kind == "template":
            pieces = _split_template(body, attrs, chunk_size, chunk_overlap)
        else:
            pieces = _split_style(body, chunk_size, chunk_overlap)

        for piece in pieces:
            if kind == "script":
                header = (
                    f"// {source} — {phrase} [{kind}]" if phrase
                    else f"// {source} [{kind}]"
                )
            else:
                header = (
                    f"<!-- {source} — {phrase} [{kind}] -->" if phrase
                    else f"<!-- {source} [{kind}] -->"
                )
            breadcrumb = {"path": source, "symbol": phrase, "block": kind}
            out.append({
                "text": header + "\n" + piece,
                "metadata": {
                    "source": source,
                    "strategy": strategy,
                    "chunk_index": idx,
                    "breadcrumb": breadcrumb,
                    "block_type": kind,
                },
            })
            idx += 1
    return out


VUE_WINNER = "E_vue_expanded_breadcrumb"
