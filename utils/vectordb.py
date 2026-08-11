import hashlib
import logging
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import get_runtime_config


_client = None

# Verify if qdrant collection exist
def check_collection_exists(name):
    return _get_qdrant_client().collection_exists(name)

# Create a Qdrant collection
def create_collection(name, vector_size):
    _get_qdrant_client().create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

# Remove a Qdrant collection
def remove_collection(name):
    _get_qdrant_client().delete_collection(name)


# Return the number of indexed points in the code collection
def get_code_collection_length():
    client = _get_qdrant_client()
    collection_name = get_runtime_config().qdrant_collection_code
    return client.count(collection_name=collection_name).count

# Get last ingestion timestamp from the metadata collection
def get_last_ingestion():
    client = _get_qdrant_client()
    collection_name = get_runtime_config().qdrant_collection_metadata

    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=[get_runtime_config().qdrant_last_ingestion_key],
        with_vectors=False,
    )

    value = records[0].payload.get(get_runtime_config().qdrant_last_ingestion_key) if records else None
    return str(value) if value else None


# Read the stored embedding model and chunking version, or None if unset.
def get_indexed_config():
    client = _get_qdrant_client()
    collection_name = get_runtime_config().qdrant_collection_metadata

    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=["embedding_model", "chunking_version"],
        with_vectors=False,
    )
    if not records or "embedding_model" not in records[0].payload:
        return None
    return {
        "embedding_model": records[0].payload.get("embedding_model"),
        "chunking_version": records[0].payload.get("chunking_version"),
    }


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Create & cache the qdrant client
def _get_qdrant_client():
    global _client
    if _client is None:
        url = get_runtime_config().qdrant_url
        client = QdrantClient(url=url)
        try:
            client.get_collections()
        except Exception as e:
            logging.getLogger("knowledge.vectordb").error(
                "cannot reach Qdrant at %s: %s", url, e)
            sys.exit(1)
        _client = client
    return _client




























# Create the collection (cosine distance) if it does not exist yet.
def ensure_collection(vector_size, recreate=False):
    client = _get_qdrant_client()
    name = get_runtime_config().qdrant_collection_code
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
    client = _get_qdrant_client()
    name = get_runtime_config().qdrant_collection_code
    records = [_to_record(c, v) for c, v in zip(chunks, vectors)]
    for start in range(0, len(records), batch_size):
        client.upsert(
            collection_name=name, points=records[start:start + batch_size])
    return len(records)


# Return the entries nearest a query vector, as read_payload result dicts.
# Empty list if the collection does not exist yet (corpus never ingested),
# so callers can surface a "run ingestion" hint instead of a 500.
def search(vector, top_k=5):
    client = _get_qdrant_client()
    name = get_runtime_config().qdrant_collection_code
    if not client.collection_exists(name):
        return []
    hits = client.query_points(
        collection_name=name, query=vector, limit=top_k,
        with_payload=True).points
    return [read_payload(hit.payload, hit.score) for hit in hits]





# Yield the payload of every stored entry, paging through the collection.
# Used to rebuild the indexed-file state; yields nothing if the collection
# does not exist yet (first run).
def iter_payloads(page_size=256):
    client = _get_qdrant_client()
    name = get_runtime_config().qdrant_collection_code
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


# Create the metadata collection (a tiny side collection that stores the
# last-ingestion timestamp) if it does not exist yet. 
def ensure_metadata_collection(collection_name):
    client = _get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )




# Persist the last-ingestion timestamp and the indexing config used.
def set_last_ingestion(collection_name, timestamp, embedding_model, chunking_version):
    ensure_metadata_collection(collection_name)
    _get_qdrant_client().upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "indexing_state")),
            vector=[0.0],
            payload={
                get_runtime_config().qdrant_last_ingestion_key: timestamp,
                "embedding_model": embedding_model,
                "chunking_version": chunking_version,
            },
        )],
    )


# Delete all indexed chunks for one repo-relative file so a reindex does not
# leave stale chunks behind when the file content or chunk count changed.
def delete_file(repository, source_path):
    client = _get_qdrant_client()
    name = get_runtime_config().qdrant_collection_code
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(
                key="repository",
                match=MatchValue(value=repository),
            ),
            FieldCondition(
                key="source_path",
                match=MatchValue(value=source_path),
            ),
        ])),
    )


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





# The file extension without the dot, e.g. "map/x.vue" -> "vue".
def _file_type(source_path):
    return Path(source_path).suffix.lower().lstrip(".")


# Hex SHA-1 of a text, used in the entry id and for change detection.
def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
