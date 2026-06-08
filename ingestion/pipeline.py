"""Ingestion pipeline for the knowledge RAG.

Turns the local Kalisio repositories into searchable vectors in the Qdrant
collection the API queries. The flow is:

    select files  ->  read & chunk  ->  enrich  ->  embed  ->  upsert

The front half (select files, read and chunk them, tag each chunk with its
repository) is implemented here; the caller passes the repositories to
index. Recent-commit enrichment, embedding and the upsert to Qdrant are
wired in next, as git history, the embedding model and the vector DB become
available.
"""

from pathlib import Path

from ingestion.parser import scan
from ingestion.chunks.md import chunk_markdown
from ingestion.chunks.js import chunk_js
from ingestion.chunks.vue import chunk_vue
from ingestion.chunks.json import chunk_json


# Which chunker handles each indexable file suffix.
CHUNKERS = {
    ".md": chunk_markdown,
    ".js": chunk_js,
    ".mjs": chunk_js,
    ".cjs": chunk_js,
    ".vue": chunk_vue,
    ".json": chunk_json,
}


# Run the ingestion job over the given repository directories.
def run(repo_dirs):
    chunks = chunk_repositories(repo_dirs)

    # Wired in next, as each piece lands:
    #   load IngestionConfig for the Qdrant collection and embedding model;
    #   enrich each chunk with its recent commits (ingestion.git_history);
    #   vectors = embeddings.encode([c["text"] for c in chunks]);
    #   vectordb.upsert(collection, chunks, vectors).
    return chunks


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
