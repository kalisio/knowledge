"""Production chunking for the Kalisio code-generation RAG.

Exposes three public entry points, one per file type:

- ``chunk_markdown()`` — nb02 winner (AST-Merge + Breadcrumb)
- ``chunk_js()``       — nb03 winner (Recursive JS + breadcrumb)
- ``chunk_vue()``      — nb03 winner (SFC dispatcher + expanded breadcrumb).
                          Defaults to the expanded-phrase variant (strategy E);
                          ``strategy="D_vue_breadcrumb"`` selects the plain
                          path-only breadcrumb.

Plus ``chunk_files()`` which dispatches by extension.

Every function returns ``list[dict]`` with keys ``text`` and ``metadata``.
``text`` includes a breadcrumb prefix (for embedding); ``metadata`` carries
the same info as structured fields (for retriever / reranker / UI).

Metadata schema (all file types)::

    Required:
        source       — rel_path of the original file
        strategy     — strategy tag, e.g. "D_js_breadcrumb"
        chunk_index  — 0-based position within the file
        breadcrumb   — structured dict:
                         {"path": str, "symbol": str, "block": str}
    Optional:
        block_type   — Vue only: "template" / "script" / "style"
        doc_title    — Markdown only: first h1 title
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_doc_title(text: str, fallback: str) -> str:
    """Return the first h1 title, or the fallback if none is present."""
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return fallback


def heading_key(metadata: dict) -> str:
    """Build a section key from EMSTS metadata for grouping adjacent atoms."""
    parts = []
    for h in ("Header 1", "Header 2", "Header 3", "Header 4"):
        if metadata.get(h):
            parts.append(metadata[h])
    return " > ".join(parts) if parts else "_root_"


def _merge_atoms(atoms, chunk_size: int) -> list[dict]:
    """Merge EMSTS atoms under the same heading section until chunk_size.

    A single atom larger than chunk_size (typically a big code block) is kept
    whole rather than cut — code integrity is the whole point of this strategy.
    """
    if not atoms:
        return []

    merged: list[dict] = []
    current_text = ""
    current_meta = atoms[0].metadata.copy()
    current_key = heading_key(atoms[0].metadata)

    for atom in atoms:
        atom_key = heading_key(atom.metadata)
        atom_text = atom.page_content.strip()
        if not atom_text:
            continue

        fits = (
            atom_key == current_key
            and len(current_text) + len(atom_text) + 2 <= chunk_size
        )
        if fits:
            current_text = (current_text + "\n\n" + atom_text).strip()
            for k, v in atom.metadata.items():
                if k == "Code" and v:
                    current_meta.setdefault("has_code", True)
                if k not in current_meta:
                    current_meta[k] = v
        else:
            if current_text:
                merged.append({"text": current_text, "metadata": current_meta.copy()})
            current_text = atom_text
            current_meta = atom.metadata.copy()
            current_key = atom_key

    if current_text:
        merged.append({"text": current_text, "metadata": current_meta.copy()})
    return merged


# ── Strategy implementations ─────────────────────────────────────────────────

def chunk_rct(
    text: str,
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Strategy A — RecursiveCharacterTextSplitter baseline."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    title = get_doc_title(text, Path(source).stem)
    docs = splitter.create_documents(
        [text], metadatas=[{"source": source, "doc_title": title}]
    )
    return [
        {
            "text": d.page_content,
            "metadata": {**d.metadata, "strategy": "A_rct", "chunk_index": i},
        }
        for i, d in enumerate(docs)
    ]


def chunk_mhs_rct(
    text: str,
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Strategy B — MarkdownHeaderTextSplitter → RCT re-split."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    re_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    title = get_doc_title(text, Path(source).stem)
    chunks: list[dict] = []
    i = 0
    for hdoc in header_splitter.split_text(text):
        for sub in re_splitter.split_documents([hdoc]):
            md = dict(sub.metadata)
            md["source"] = source
            md["doc_title"] = title
            md["strategy"] = "B_mhs_rct"
            md["chunk_index"] = i
            chunks.append({"text": sub.page_content, "metadata": md})
            i += 1
    return chunks


def chunk_ast_merge(
    text: str,
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict]:
    """Strategy C — EMSTS atoms merged under same heading (structural integrity)."""
    splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    atoms = splitter.split_text(text)
    merged = _merge_atoms(atoms, chunk_size=chunk_size)
    title = get_doc_title(text, Path(source).stem)
    for i, chunk in enumerate(merged):
        chunk["metadata"]["source"] = source
        chunk["metadata"]["doc_title"] = title
        chunk["metadata"]["strategy"] = "C_ast_merge"
        chunk["metadata"]["chunk_index"] = i
    return merged


def chunk_ast_breadcrumb(
    text: str,
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict]:
    """Strategy D — AST-Merge + breadcrumb prefix injected into chunk text.

    The breadcrumb (`Context: h1 > h2 > h3`) gives the embedding model and
    downstream LLM an explicit hierarchy without forcing them to read metadata.
    This is the nb02 winner for `title` and `section` queries; nb02 Part 5 also
    notes the known tradeoff on pure `symbol` queries.
    """
    splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    atoms = splitter.split_text(text)
    merged = _merge_atoms(atoms, chunk_size=chunk_size)
    title = get_doc_title(text, Path(source).stem)
    for i, chunk in enumerate(merged):
        path_parts = [
            chunk["metadata"][h]
            for h in ("Header 1", "Header 2", "Header 3", "Header 4")
            if chunk["metadata"].get(h)
        ]
        breadcrumb = " > ".join(path_parts) if path_parts else title
        prefix = f"Context: {breadcrumb}\nSource: {source}\n\n"
        chunk["text"] = prefix + chunk["text"]
        chunk["metadata"]["source"] = source
        chunk["metadata"]["doc_title"] = title
        chunk["metadata"]["strategy"] = "D_ast_breadcrumb"
        chunk["metadata"]["breadcrumb"] = breadcrumb
        chunk["metadata"]["chunk_index"] = i
    return merged


# ── JS chunking (nb03 winner: D — Recursive JS + breadcrumb) ────────────────

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
    """Chunk one JS file using the nb03 winner strategy (D — breadcrumb)."""
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


# ── Vue SFC chunking (nb03 winner: E — SFC dispatcher + expanded breadcrumb) ─

_SFC_BLOCK_RE = re.compile(
    r"<(template|script|style)\b([^>]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_LANG_ATTR_RE = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

_VUE_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _component_phrase(source: str) -> str:
    """Derive a human-readable phrase from a ``.vue`` filename.

    ``KZoomControl.vue`` → ``"zoom control"``.  Consecutive capitals are
    kept as one acronym token (``KHTTPClient`` → ``"http client"``); the
    leading ``K`` prefix and 1-char shards are dropped.
    """
    stem = Path(source).stem
    name = stem[1:] if stem.startswith("K") and len(stem) > 1 else stem
    words = [w.lower() for w in _VUE_CAMEL_RE.split(name) if w]
    words = [w for w in words if len(w) > 1]
    return " ".join(words)


def _vue_split_template(body: str, attrs: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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


def _vue_split_style(body: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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

    Two strategies are available:

    - ``E_vue_expanded_breadcrumb`` (default, nb03 winner) — SFC dispatcher
      with a breadcrumb that includes both the file path and the component's
      paraphrased name, so dense retrieval sees surface tokens shared with
      natural-language queries (``"zoom control"`` matches ``KZoomControl``).
    - ``D_vue_breadcrumb`` — path-only breadcrumb, retained for callers
      that prefer minimal per-chunk overhead.

    The ``F_hybrid`` winner from nb03 uses E chunks at retrieval time
    (dense + BM25 fused with RRF); it is not a chunking strategy.
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
            pieces = _vue_split_template(body, attrs, chunk_size, chunk_overlap)
        else:
            pieces = _vue_split_style(body, chunk_size, chunk_overlap)

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
            breadcrumb = {
                "path": source, "symbol": phrase, "block": kind,
            }
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


# ── Public API ───────────────────────────────────────────────────────────────

MD_STRATEGIES = {
    "A_rct": chunk_rct,
    "B_mhs_rct": chunk_mhs_rct,
    "C_ast_merge": chunk_ast_merge,
    "D_ast_breadcrumb": chunk_ast_breadcrumb,
}

STRATEGIES = MD_STRATEGIES

MD_WINNER = "D_ast_breadcrumb"
JS_WINNER = "D_js_breadcrumb"
VUE_WINNER = "E_vue_expanded_breadcrumb"

# Legacy alias
WINNER_STRATEGY = MD_WINNER


def chunk_markdown(
    text: str,
    source: str,
    *,
    strategy: str = MD_WINNER,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk one markdown document using the selected strategy."""
    if strategy not in MD_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from {list(MD_STRATEGIES)}"
        )
    func = MD_STRATEGIES[strategy]
    if strategy in ("A_rct", "B_mhs_rct"):
        return func(text, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return func(text, source, chunk_size=chunk_size)


_EXT_DISPATCH = {
    ".md": "markdown",
    ".js": "js",
    ".mjs": "js",
    ".vue": "vue",
}


def chunk_files(
    files: Iterable,
    *,
    md_strategy: str = MD_WINNER,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk a batch of corpus_filter records, dispatching by file extension.

    Accepts any iterable of objects exposing ``.path`` and ``.rel_path``.
    """
    out: list[dict] = []
    for rec in files:
        text = rec.path.read_text(encoding="utf-8", errors="ignore")
        src = str(rec.rel_path)
        ext = rec.path.suffix.lower()
        kind = _EXT_DISPATCH.get(ext)
        if kind == "markdown":
            out.extend(chunk_markdown(
                text, src, strategy=md_strategy,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            ))
        elif kind == "js":
            out.extend(chunk_js(text, src))
        elif kind == "vue":
            out.extend(chunk_vue(text, src))
    return out
