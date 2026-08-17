from ingestion.chunkers.markdown import chunk_markdown
from ingestion.chunkers.javascript import chunk_javascript
from ingestion.chunkers.vue import chunk_vue
from ingestion.chunkers.json import chunk_json
from ingestion.indexed_file_state import compute_file_sha1, get_file_key

# Bump by hand when a chunker's splitting behaviour changes.
CHUNKING_VERSION = 1

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
    for path in files:
        chunker = _CHUNKERS.get(path.suffix.lower())
        if chunker is None:
            continue
        repository, source_path = get_file_key(path, workspace)
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_sha1 = compute_file_sha1(text)
        for chunk in chunker(text, source_path):
            chunk["metadata"]["repository"] = repository
            chunk["metadata"]["file_sha1"] = file_sha1
            chunks.append(chunk)
    return chunks
