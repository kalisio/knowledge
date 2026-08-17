from pathlib import Path

from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter as MarkdownSyntaxTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Target chunk length and overlap in characters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

_HEADING_LEVELS = ("Header 1", "Header 2", "Header 3", "Header 4")


# Chunk one markdown file: merge atoms per heading, prefix a breadcrumb.
def chunk_markdown(text, source_path):
    atoms = MarkdownSyntaxTextSplitter(strip_headers=False).split_text(text)
    title = _doc_title(text, Path(source_path).stem)
    sections = _split_prose(_merge_by_heading(atoms))
    chunks = []
    for chunk_index, section in enumerate(sections):
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

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

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


# Split the prose sections a single oversized atom left above CHUNK_SIZE.
# Without this a heading with no subheading is never cut, so a long document
# reaches the embedding model as one vector and is silently truncated there.
def _split_prose(sections):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split = []
    for section in sections:
        if _stays_whole(section):
            split.append(section)
            continue
        split += [{"text": piece, "metadata": section["metadata"]}
                  for piece in splitter.split_text(section["text"])]
    return split


# A section is left alone when it already fits, or when it is the fenced code
# the size exemption in _merge_by_heading is there to protect; the splitter
# marks such an atom with a Code entry naming the language.
def _stays_whole(section):
    return len(section["text"]) <= CHUNK_SIZE or "Code" in section["metadata"]
