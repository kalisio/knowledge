"""Qdrant access plus the shared payload schema for ingest and search.

Talks to the Qdrant instance at QDRANT_URL on the collection named by
QDRANT_COLLECTION: create the collection, upsert, search, and scroll it.
The payload (the fields stored alongside each vector) and the deterministic
id are defined here too -- build_payload / read_payload / payload_id -- so
the ingestion job (writer) and the API (reader) share one shape and one id
scheme. The id is derived from the chunk, so re-indexing the same chunk
overwrites its entry instead of duplicating it.
"""

import hashlib
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_runtime_config


_client = None


# Create the collection (cosine distance) if it does not exist yet.
def ensure_collection(vector_size, recreate=False):
    client = _get_client()
    name = get_runtime_config().qdrant_collection
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE),
        )


# Upsert chunks with their vectors. Same chunk -> same id, so a re-run
# overwrites its entry instead of creating a duplicate.
def upsert(chunks, vectors, batch_size=64):
    client = _get_client()
    name = get_runtime_config().qdrant_collection
    records = [_to_record(c, v) for c, v in zip(chunks, vectors)]
    for start in range(0, len(records), batch_size):
        client.upsert(
            collection_name=name, points=records[start:start + batch_size])
    return len(records)


# Return the entries nearest a query vector, as read_payload result dicts.
# Empty list if the collection does not exist yet (corpus never ingested),
# so callers can surface a "run ingestion" hint instead of a 500.
def search(vector, top_k=5):
    client = _get_client()
    name = get_runtime_config().qdrant_collection
    if not client.collection_exists(name):
        return []
    hits = client.query_points(
        collection_name=name, query=vector, limit=top_k,
        with_payload=True).points
    return [read_payload(hit.payload, hit.score) for hit in hits]


# Number of indexed entries, or 0 when the collection does not exist yet.
# Used to tell "index not built" apart from "query had no match".
def count():
    client = _get_client()
    name = get_runtime_config().qdrant_collection
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name).count


# Yield the payload of every stored entry, paging through the collection.
# Used to rebuild the indexed manifest; yields nothing if the collection
# does not exist yet (first run).
def iter_payloads(page_size=256):
    client = _get_client()
    name = get_runtime_config().qdrant_collection
    if not client.collection_exists(name):
        return
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name, with_payload=True, with_vectors=False,
            limit=page_size, offset=offset)
        for record in records:
            yield record.payload
        if offset is None:
            break


# Deterministic id for a chunk's stored entry: same chunk -> same id
# (idempotent upsert). Includes the repository since source_path is
# repo-relative.
def payload_id(repository, source_path, chunk_index, text):
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


# Reconstruct a search result from a stored payload and similarity score.
# The API calls this to turn Qdrant hits into answer sources.
def read_payload(payload, score):
    return {
        "source_path": payload.get("source_path", ""),
        "repository": payload.get("repository", ""),
        "breadcrumb": payload.get("breadcrumb", ""),
        "chunk_index": payload.get("chunk_index", 0),
        "content": payload.get("text", ""),
        "commit_history": payload.get("commit_history", []),
        "score": score,
    }


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Assemble the id + vector + payload into the PointStruct we upsert.
def _to_record(chunk, vector):
    meta = chunk["metadata"]
    entry_id = payload_id(
        meta.get("repository", ""), meta["source_path"],
        meta["chunk_index"], chunk["text"])
    return PointStruct(
        id=entry_id, vector=vector, payload=build_payload(chunk))


# Lazily create and cache the Qdrant client.
def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=get_runtime_config().qdrant_url)
    return _client


# The file extension without the dot, e.g. "map/x.vue" -> "vue".
def _file_type(source_path):
    return Path(source_path).suffix.lower().lstrip(".")


# Hex SHA-1 of a text, used in the entry id and for change detection.
def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
