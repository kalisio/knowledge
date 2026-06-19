"""Split a Markdown file into chunks for indexing.

The file is broken at its Markdown headings (``#``, ``##``, ``###`` …), and
consecutive pieces under the same heading are joined until they reach about
CHUNK_SIZE characters. A piece already larger than that (typically a long
fenced code block) is kept whole instead of being cut in the middle.

Each chunk's text starts with its heading path and source file, e.g.
``Context: Title > Section`` / ``Source: <file>``, so the chunk is
self-describing once embedded and later shown to the model.

Returns a list of ``{"text", "metadata"}`` dicts: ``text`` is the content to
embed (with that prefix); ``metadata`` repeats the same context as separate
fields. Repository and commit info is added later, before the chunk is stored
in the vector database.
"""

from pathlib import Path

from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter


# Target chunk length in characters; sections are joined up to this size.
CHUNK_SIZE = 500

# Heading levels the splitter labels each piece with, from outermost in.
_HEADING_LEVELS = ("Header 1", "Header 2", "Header 3", "Header 4")


# Split one Markdown file into heading sections, each with a context prefix.
def chunk_markdown(text, source_path, chunk_size=CHUNK_SIZE):
    splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    pieces = splitter.split_text(text)
    title = _doc_title(text, Path(source_path).stem)

    chunks = []
    for index, section in enumerate(_merge_by_heading(pieces, chunk_size)):
        breadcrumb = _breadcrumb(section["metadata"], title)
        prefix = f"Context: {breadcrumb}\nSource: {source_path}\n\n"
        chunks.append({
            "text": prefix + section["text"],
            "metadata": {
                "source_path": source_path,
                "doc_title": title,
                "breadcrumb": breadcrumb,
                "chunk_index": index,
            },
        })
    return chunks


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Return the file's first top-level heading ("# Title"), or the fallback.
def _doc_title(text, fallback):
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return fallback


# Join the headings a section sits under into "Title > Section > ...".
def _breadcrumb(metadata, fallback):
    parts = [metadata[h] for h in _HEADING_LEVELS if metadata.get(h)]
    return " > ".join(parts) if parts else fallback


# Join consecutive same-heading pieces up to chunk_size; keep big ones whole.
def _merge_by_heading(pieces, chunk_size):
    sections = []
    current_text = ""
    current_meta = {}
    current_key = None

    for piece in pieces:
        piece_text = piece.page_content.strip()
        if not piece_text:
            continue
        key = _heading_key(piece.metadata)
        fits = (
            key == current_key
            and len(current_text) + len(piece_text) + 2 <= chunk_size
        )
        if fits:
            current_text = f"{current_text}\n\n{piece_text}"
        else:
            if current_text:
                sections.append(
                    {"text": current_text, "metadata": current_meta})
            current_text = piece_text
            current_meta = dict(piece.metadata)
            current_key = key

    if current_text:
        sections.append({"text": current_text, "metadata": current_meta})
    return sections


# Identify which heading section a piece belongs to, for grouping.
def _heading_key(metadata):
    parts = [metadata[h] for h in _HEADING_LEVELS if metadata.get(h)]
    return " > ".join(parts) if parts else "_root_"
