"""Routes each file to the chunker that knows its format."""

from ingestion.chunkers.markdown import chunk_markdown
from ingestion.chunkers.javascript import chunk_javascript
from ingestion.chunkers.vue import chunk_vue
from ingestion.chunkers.json import chunk_json
from ingestion.pipeline.change_detection import compute_file_sha1, get_file_key

# Bump by hand when a chunker's splitting behaviour changes, or when the
# stored payload does. 2: chunks carry their line range and the payload moved
# to the path/repo/content contract. 3: the commit history left the chunk
# payload for a per-file entry. 4: a chunk is located by its whole run of
# lines, so the stored line ranges change.
CHUNKING_VERSION = 4

# Which chunker handles each file extension.
_CHUNKERS = {
    ".md": chunk_markdown,
    ".js": chunk_javascript,
    ".mjs": chunk_javascript,
    ".cjs": chunk_javascript,
    ".vue": chunk_vue,
    ".json": chunk_json,
}


# Chunk each file into tagged chunks, stamped with the file-wide metadata
# the chunkers know nothing about: its repository and its content digest.
def chunk_files(files, workspace):
    chunks = []
    for file_path in files:
        chunker = _CHUNKERS.get(file_path.suffix.lower())
        if chunker is None:
            continue
        repository, path = get_file_key(file_path, workspace)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_sha1 = compute_file_sha1(text)
        for chunk in chunker(text, path):
            chunk["metadata"]["repo"] = repository
            chunk["metadata"]["file_sha1"] = file_sha1
            chunks.append(chunk)
    return chunks
