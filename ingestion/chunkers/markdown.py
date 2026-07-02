from pathlib import Path

from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter as MarkdownSyntaxTextSplitter,
)

# Target chunk length in characters.
CHUNK_SIZE = 500

_HEADING_LEVELS = ("Header 1", "Header 2", "Header 3", "Header 4")


# Chunk one markdown file: merge atoms per heading, prefix a breadcrumb.
def chunk_markdown(text, source_path):
    atoms = MarkdownSyntaxTextSplitter(strip_headers=False).split_text(text)
    title = _doc_title(text, Path(source_path).stem)
    chunks = []
    for chunk_index, section in enumerate(_merge_by_heading(atoms)):
        breadcrumb = _breadcrumb(section["metadata"], title)
        prefix = f"Context: {breadcrumb}\nSource: {source_path}\n\n"
        chunks.append({
            "text": prefix + section["text"],
            "metadata": {
                "source_path": source_path,
                "chunk_index": chunk_index,
                "breadcrumb": breadcrumb,
            },
        })
    return chunks


# Get the file's first h1 title, or `fallback` when it has none.
def _doc_title(text, fallback):
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return fallback


# Join a section's headings into "h1 > h2 > h3", or `fallback` at the root.
def _breadcrumb(metadata, fallback):
    parts = [metadata[level] for level in _HEADING_LEVELS if metadata.get(level)]
    return " > ".join(parts) if parts else fallback


# Merge adjacent atoms sharing a heading up to CHUNK_SIZE; an atom larger
# than that (usually a code block) stays whole so code is never split.
def _merge_by_heading(atoms):
    sections = []
    text = ""
    metadata = {}
    key = None
    for atom in atoms:
        atom_text = atom.page_content.strip()
        if not atom_text:
            continue
        atom_key = _breadcrumb(atom.metadata, "_root_")
        if atom_key == key and len(text) + len(atom_text) + 2 <= CHUNK_SIZE:
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
