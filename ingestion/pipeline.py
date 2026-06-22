"""Ingestion pipeline for the knowledge RAG.

Turns the local Kalisio repositories into searchable vectors in the Qdrant
collection the API queries. The flow is:

    select files  ->  read & chunk  ->  enrich  ->  embed  ->  upsert

Files are selected and chunked, each chunk tagged with its repository and
whole-file hash. Chunks of files already indexed at the same hash are
dropped; the rest are enriched with the file's recent commit subjects,
embedded with the configured model, and upserted into Qdrant. The caller
passes the repositories to index.
"""

from pathlib import Path

from ingestion.parser import scan
from ingestion.chunks.md import chunk_markdown
from ingestion.chunks.js import chunk_js
from ingestion.chunks.vue import chunk_vue
from ingestion.chunks.json import chunk_json
from ingestion import manifest
from ingestion import git_history
import utils.embeddings as embeddings
import utils.vectordb as vectordb


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
# With incremental=True, chunks of unchanged files (same file hash as
# already indexed) are dropped before the expensive embedding step.
def run(repo_dirs, incremental=True, candidate_files=None):
    chunks = _chunk_repositories(repo_dirs, candidate_files=candidate_files)
    if incremental:
        chunks = manifest.select_changed(chunks, manifest.load())
    if not chunks:
        return 0
    _delete_existing(chunks)
    _enrich_commits(chunks, repo_dirs)
    vectors = embeddings.encode_batch([chunk["text"] for chunk in chunks])
    vectordb.ensure_collection(len(vectors[0]))
    return vectordb.upsert(chunks, vectors)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Attach each file's recent commit subjects to its chunks. Cached per file
# (a file's chunks share one history) and run only on the chunks about to
# be embedded, so unchanged files cost no git calls.
def _enrich_commits(chunks, repo_dirs):
    repo_paths = {Path(d).name: Path(d) for d in repo_dirs}
    history = {}
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["repository"], meta["source_path"])
        if key not in history:
            repo_dir = repo_paths.get(meta["repository"])
            history[key] = (
                git_history.recent_commits(repo_dir, meta["source_path"])
                if repo_dir else [])
        meta["commit_history"] = history[key]


# Chunk every indexable file across several repositories.
def _chunk_repositories(repo_dirs, candidate_files=None):
    chunks = []
    for repo_dir in repo_dirs:
        repo_dir = Path(repo_dir)
        selected = None
        if candidate_files is not None:
            selected = candidate_files.get(repo_dir.name, set())
        chunks.extend(_chunk_repo(repo_dir, candidate_files=selected))
    return chunks


# Chunk every indexable file in one repository, tagging chunks with its name.
def _chunk_repo(repo_dir, candidate_files=None):
    repo_dir = Path(repo_dir)
    repository = repo_dir.name
    chunks = []
    for path in scan(repo_dir):
        chunker = CHUNKERS.get(path.suffix.lower())
        if chunker is None:
            continue
        source_path = str(path.relative_to(repo_dir))
        if candidate_files is not None and source_path not in candidate_files:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        digest = manifest.file_sha1(text)
        for chunk in chunker(text, source_path):
            chunk["metadata"]["repository"] = repository
            chunk["metadata"]["file_sha1"] = digest
            chunks.append(chunk)
    return chunks


# Delete all existing chunks for the files about to be reindexed so changed
# files do not leave stale chunks behind when the chunk content/count changes.
def _delete_existing(chunks):
    seen = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["repository"], meta["source_path"])
        if key in seen:
            continue
        vectordb.delete_file(*key)
        seen.add(key)
