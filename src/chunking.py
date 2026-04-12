"""Production chunking for the Kalisio code-generation RAG.

This module exposes the winner strategy selected in nb02
(AST-Merge + Breadcrumb) plus the supporting helpers that were originally
defined inside the notebook. Extracting them here means:

- The notebook stays the authoritative benchmark but reuses these helpers
  via import, so there is only one source of truth.
- The parameter sweep in Part 6 and the code-generation experiment in
  Part 7 can run as plain scripts without executing the whole notebook.
- Downstream pipelines (nb03 multi-type chunking, and future indexing
  jobs) import `chunk_markdown` directly.

The winner selection itself is still made by nb02's Part 5 on reproducible
metrics — this module just packages the chosen implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
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
            "metadata": {**d.metadata, "strategy": "A", "chunk_index": i},
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
            md["strategy"] = "B"
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
        chunk["metadata"]["strategy"] = "C"
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
        chunk["metadata"]["strategy"] = "D"
        chunk["metadata"]["breadcrumb"] = breadcrumb
        chunk["metadata"]["chunk_index"] = i
    return merged


# ── Public API ───────────────────────────────────────────────────────────────

STRATEGIES = {
    "A_rct": chunk_rct,
    "B_mhs_rct": chunk_mhs_rct,
    "C_ast_merge": chunk_ast_merge,
    "D_ast_breadcrumb": chunk_ast_breadcrumb,
}

# nb02 winner. Change this only after re-running the deterministic benchmark.
WINNER_STRATEGY = "D_ast_breadcrumb"


def chunk_markdown(
    text: str,
    source: str,
    *,
    strategy: str = WINNER_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk one markdown document using the selected strategy.

    Each returned item is a dict with `text` and `metadata` keys, ready to be
    embedded and upserted into a vector store.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from {list(STRATEGIES)}"
        )
    func = STRATEGIES[strategy]
    if strategy in ("A_rct", "B_mhs_rct"):
        return func(text, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return func(text, source, chunk_size=chunk_size)


def chunk_files(
    files: Iterable,
    *,
    strategy: str = WINNER_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk a batch of corpus_filter records. Accepts any iterable of objects
    exposing `.path` and `.rel_path`."""
    out: list[dict] = []
    for rec in files:
        text = rec.path.read_text(encoding="utf-8")
        out.extend(
            chunk_markdown(
                text,
                source=str(rec.rel_path),
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return out
