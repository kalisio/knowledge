"""Qdrant point payload: the shared schema for ingest and search.

A Qdrant point is (id, vector, payload). This module owns the payload — the
fields stored alongside the vector — so the ingestion job (which writes
points) and the API (which reads them back) agree on one shape. The id is
derived deterministically from the chunk, so re-indexing the same chunk
overwrites its point instead of creating a duplicate.
"""

import hashlib
import uuid
from pathlib import Path


# Deterministic point id from a chunk: same chunk -> same id (idempotent
# upsert). Includes the repository since source_path is repo-relative.
def point_id(repository, source_path, chunk_index, text):
    key = f"{repository}:{source_path}:{chunk_index}:{_sha1(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# Build the payload stored alongside a chunk's vector (everything but the
# vector). The ingestion job calls this before upserting to Qdrant.
def build_payload(chunk):
    metadata = chunk["metadata"]
    source_path = metadata["source_path"]
    return {
        "text": chunk["text"],
        "source_path": source_path,
        "repository": metadata.get("repository", ""),
        "file_type": _file_type(source_path),
        "chunk_index": metadata["chunk_index"],
        "breadcrumb": metadata.get("breadcrumb", ""),
        "commit_history": metadata.get("commit_history", []),
        "text_sha1": _sha1(chunk["text"]),
        "file_sha1": metadata.get("file_sha1", ""),
    }


# Reconstruct a search result from a point's payload and similarity score.
# The API calls this to turn Qdrant hits into answer sources.
def read_point(payload, score):
    return {
        "source_path": payload.get("source_path", ""),
        "repository": payload.get("repository", ""),
        "breadcrumb": payload.get("breadcrumb", ""),
        "chunk_index": payload.get("chunk_index", 0),
        "content": payload.get("text", ""),
        "commit_history": payload.get("commit_history", []),
        "score": score,
    }


# The file extension without the dot, e.g. "map/x.vue" -> "vue".
def _file_type(source_path):
    return Path(source_path).suffix.lower().lstrip(".")


# Hex SHA-1 of a text, used in the point id and for change detection.
def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
