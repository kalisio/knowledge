"""Cuts a markdown file into chunks, one per heading section."""

from pathlib import Path

from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter as MarkdownSyntaxTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ingestion.chunkers.line_locator import LineLocator
from ingestion.config import get_config

_HEADING_LEVELS = ("Header 1", "Header 2", "Header 3", "Header 4")


# Chunk one markdown file: merge atoms per heading, prefix a breadcrumb.
def chunk_markdown(text, path):
    config = get_config()
    atoms = MarkdownSyntaxTextSplitter(strip_headers=False).split_text(text)
    title = _doc_title(text, Path(path).stem)
    sections = _split_prose(_merge_by_heading(atoms, config.chunk_size),
                            config)
    locator = LineLocator(text)
    chunks = []
    for chunk_index, section in enumerate(sections):
        breadcrumb = _breadcrumb(section["metadata"], title)
        prefix = f"Context: {breadcrumb}\nSource: {path}\n\n"
        start_line, end_line = (locator.locate(section["text"])
                                or locator.whole_file())
        chunks.append({
            "text": prefix + section["text"],
            "metadata": {
                "path": path,
                "chunk_index": chunk_index,
                "breadcrumb": breadcrumb,
                "start_line": start_line,
                "end_line": end_line,
            },
        })
    return chunks

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

# Get the file's first h1 title, or `fallback` when it has none.
def _doc_title(text, fallback):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# Join a section's headings into "h1 > h2 > h3", or `fallback` at the root.
def _breadcrumb(metadata, fallback):
    parts = [metadata[level] for level in _HEADING_LEVELS if metadata.get(level)]
    return " > ".join(parts) if parts else fallback


# Merge adjacent atoms sharing a heading up to `chunk_size`; an atom larger
# than that (usually a code block) stays whole so code is never split.
def _merge_by_heading(atoms, chunk_size):
    sections = []
    text = ""
    metadata = {}
    key = None
    for atom in atoms:
        atom_text = atom.page_content.strip()
        if not atom_text:
            continue
        atom_key = _breadcrumb(atom.metadata, "_root_")
        if atom_key == key and len(text) + len(atom_text) + 2 <= chunk_size:
            text = f"{text}\n\n{atom_text}"
        else:
            if text:
                sections.append({"text": text, "metadata": metadata})
            text = atom_text
            metadata = dict(atom.metadata)
            key = atom_key
    if text:
        sections.append({"text": text, "metadata": metadata})
    return sections


# Split the prose sections a single oversized atom left above chunk_size.
# Without this a heading with no subheading is never cut, so a long document
# reaches the embedding model as one vector and is silently truncated there.
def _split_prose(sections, config):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    split = []
    for section in sections:
        if _stays_whole(section, config.chunk_size):
            split.append(section)
            continue
        split += [{"text": piece, "metadata": section["metadata"]}
                  for piece in splitter.split_text(section["text"])]
    return split


# A section is left alone when it already fits, or when it is the fenced code
# the size exemption in _merge_by_heading is there to protect; the splitter
# marks such an atom with a Code entry naming the language.
def _stays_whole(section, chunk_size):
    return len(section["text"]) <= chunk_size or "Code" in section["metadata"]
