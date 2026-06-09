"""Ingestion pipeline for the knowledge RAG.

Turns the local Kalisio repositories into searchable vectors in the Qdrant
collection the API queries. The flow is:

    select files  ->  read & chunk  ->  enrich  ->  embed  ->  upsert

Files are selected and chunked, each chunk tagged with its repository, then
embedded with the configured model and upserted into Qdrant. The caller
passes the repositories to index. Recent-commit enrichment is still
deferred, so commit_history stays empty until ingestion.git_history lands.
"""

from pathlib import Path

from ingestion.parser import scan
from ingestion.chunks.md import chunk_markdown
from ingestion.chunks.js import chunk_js
from ingestion.chunks.vue import chunk_vue
from ingestion.chunks.json import chunk_json
import utils.embeddings.service as embeddings
import utils.vectordb.client as vectordb


# Which chunker handles each indexable file suffix.
CHUNKERS = {
    ".md": chunk_markdown,
    ".js": chunk_js,
    ".mjs": chunk_js,
    ".cjs": chunk_js,
    ".vue": chunk_vue,
    ".json": chunk_json,
}


# Run the ingestion job: chunk the repositories, embed every chunk, and
# upsert the vectors into Qdrant. Returns the number of chunks indexed.
def run(repo_dirs):
    chunks = chunk_repositories(repo_dirs)
    if not chunks:
        return 0
    vectors = embeddings.encode_batch([chunk["text"] for chunk in chunks])
    vectordb.ensure_collection(len(vectors[0]))
    return vectordb.upsert(chunks, vectors)


# Chunk every indexable file across several repositories.
def chunk_repositories(repo_dirs):
    chunks = []
    for repo_dir in repo_dirs:
        chunks.extend(chunk_repo(repo_dir))
    return chunks


# Chunk every indexable file in one repository, tagging chunks with its name.
def chunk_repo(repo_dir):
    repo_dir = Path(repo_dir)
    repository = repo_dir.name
    chunks = []
    for path in scan(repo_dir):
        chunker = CHUNKERS.get(path.suffix.lower())
        if chunker is None:
            continue
        source_path = str(path.relative_to(repo_dir))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunker(text, source_path):
            chunk["metadata"]["repository"] = repository
            chunks.append(chunk)
    return chunks
