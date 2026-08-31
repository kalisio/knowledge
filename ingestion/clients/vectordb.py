"""Writes the Kalisio corpus into Qdrant."""

import hashlib
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

from ingestion.config import get_config
from ingestion.logger import get_logger


_client = None

log = get_logger("vectordb")


# Raised when Qdrant cannot be reached at all.
class QdrantUnreachable(RuntimeError):
    pass

# Verify if qdrant collection exist
def check_collection_exists(name):
    return _get_qdrant_client().collection_exists(name)

# Vector size the collection was created with, or None when it does not exist
def get_collection_vector_size(name):
    client = _get_qdrant_client()
    if not client.collection_exists(name):
        return None
    return client.get_collection(name).config.params.vectors.size

# Create a Qdrant collection
def create_collection(name, vector_size):
    _get_qdrant_client().create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("collection '%s' created (vector_size %d)", name, vector_size)

# Remove a Qdrant collection
def remove_collection(name):
    _get_qdrant_client().delete_collection(name)
    log.info("collection '%s' removed", name)


# Return the number of indexed points in the code collection, or 0 when the
# collection does not exist yet ("index not built" vs "query had no match").
def get_code_collection_length():
    client = _get_qdrant_client()
    collection_name = get_config().qdrant_collection_code
    if not client.collection_exists(collection_name):
        return 0
    return client.count(collection_name=collection_name).count

# Get last ingestion timestamp from the metadata collection
def get_last_ingestion():
    client = _get_qdrant_client()
    collection_name = get_config().qdrant_collection_metadata

    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=[get_config().qdrant_last_ingestion_key],
        with_vectors=False,
    )

    value = records[0].payload.get(get_config().qdrant_last_ingestion_key) if records else None
    return str(value) if value else None


# Read the stored embedding model and chunking version, or None if unset.
def get_indexed_config():
    client = _get_qdrant_client()
    collection_name = get_config().qdrant_collection_metadata

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


# Create the collection (cosine distance) if it does not exist yet. An
# existing collection is dropped first when `recreate` is set, or when its
# vector size no longer matches: vectors of another dimension cannot be
# upserted into it.
def ensure_collection(name, vector_size, recreate=False):
    indexed_vector_size = get_collection_vector_size(name)
    if indexed_vector_size == vector_size and not recreate:
        return
    if indexed_vector_size is not None:
        remove_collection(name)
    create_collection(name, vector_size)


# Upsert chunks with their vectors. Same chunk -> same id, so a re-run
# overwrites its entry instead of creating a duplicate.
def upsert(chunks, vectors, batch_size=64):
    client = _get_qdrant_client()
    name = get_config().qdrant_collection_code
    records = [_to_record(c, v) for c, v in zip(chunks, vectors)]
    for start in range(0, len(records), batch_size):
        client.upsert(
            collection_name=name, points=records[start:start + batch_size])
    return len(records)


# The identity and digest of every stored chunk. Only the three fields the
# indexed-file state needs are asked for: the whole payload carries the
# chunk text, which would mean pulling the entire corpus over the wire to
# answer a question about hashes.
def iter_chunk_payloads(page_size=256):
    yield from _iter_payloads(get_config().qdrant_collection_code,
                              ["repo", "path", "file_sha1"], page_size)


# The same, from the file entries: one per scanned file, including the files
# that yield no chunk and therefore appear nowhere in the code collection.
def iter_file_entry_payloads(page_size=256):
    yield from _iter_payloads(get_config().qdrant_collection_files,
                              ["repo", "path", "file_sha1"], page_size)


# Persist the last-ingestion timestamp and the indexing config used.
def set_last_ingestion(collection_name, timestamp, embedding_model, chunking_version):
    ensure_collection(collection_name, get_config().qdrant_vector_size_collection_metadata)
    _get_qdrant_client().upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "indexing_state")),
            vector=[0.0],
            payload={
                get_config().qdrant_last_ingestion_key: timestamp,
                "embedding_model": embedding_model,
                "chunking_version": chunking_version,
            },
        )],
    )


# Store one entry per file holding its commit history and its digest.
# Keeping the history here rather than on every chunk is what makes a real
# history affordable: a file cut into twenty chunks used to carry twenty
# copies of it. The digest rides along because this is the only record a
# file that yields no chunk ever gets -- `file_hashes` maps (repo, path) to
# the digest of what was just scanned.
def upsert_file_entries(histories, file_hashes=None, batch_size=64):
    if not histories:
        return 0
    client = _get_qdrant_client()
    name = get_config().qdrant_collection_files
    file_hashes = file_hashes or {}
    points = [
        PointStruct(
            id=file_entry_id(repo, path),
            vector=[0.0],
            payload={"repo": repo, "path": path,
                     "commit_history": list(subjects),
                     "file_sha1": file_hashes.get((repo, path), "")},
        )
        for (repo, path), subjects in histories.items()
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=name,
                      points=points[start:start + batch_size])
    return len(points)


# Drop the file entry of a file that is no longer indexed.
def delete_file_entry(repository, path):
    client = _get_qdrant_client()
    name = get_config().qdrant_collection_files
    if not client.collection_exists(name):
        return
    client.delete(collection_name=name,
                  points_selector=[file_entry_id(repository, path)])


# Deterministic id of a file entry: one entry per file, overwritten in place.
def file_entry_id(repository, path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"file:{repository}:{path}"))


# Delete all indexed chunks for one repo-relative file so a reindex does not
# leave stale chunks behind when the file content or chunk count changed.
def delete_file(repository, path):
    client = _get_qdrant_client()
    name = get_config().qdrant_collection_code
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(
                key="repo",
                match=MatchValue(value=repository),
            ),
            FieldCondition(
                key="path",
                match=MatchValue(value=path),
            ),
        ])),
    )


# Deterministic id for a chunk's stored entry: same chunk -> same id
# (idempotent upsert). Includes the repository since the path is
# repo-relative.
def payload_id(repository, path, chunk_index, text):
    key = f"{repository}:{path}:{chunk_index}:{_sha1(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# api/clients/vectordb.py reads this shape back. The end-to-end suite runs
# both against one Qdrant, so a drift between them fails a test rather than
# silently returning nothing.
# Build the payload stored alongside a chunk's vector (everything but the
# vector). The commit history is NOT part of it: it belongs to the file, not
# to the chunk, and is stored once per file (see upsert_file_entries).
def build_payload(chunk):
    metadata = chunk["metadata"]
    path = metadata["path"]
    return {
        "path": path,
        "repo": metadata.get("repo", ""),
        "start_line": metadata["start_line"],
        "end_line": metadata["end_line"],
        "content": chunk["text"],
        "file_type": _file_type(path),
        "chunk_index": metadata["chunk_index"],
        "breadcrumb": metadata.get("breadcrumb", ""),
        "content_sha1": _sha1(chunk["text"]),
        "file_sha1": metadata.get("file_sha1", ""),
    }


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Create & cache the qdrant client. An unreachable Qdrant raises, so the
# caller decides what it means: the ingestion job cannot work without it, the
# API only loses its index and must still serve /health.
def _get_qdrant_client():
    global _client
    if _client is None:
        url = get_config().qdrant_url
        client = QdrantClient(url=url)
        try:
            client.get_collections()
        except Exception as exc:
            log.error("cannot reach Qdrant at %s: %s", url, exc)
            raise QdrantUnreachable(
                f"cannot reach Qdrant at {url}: {exc}") from exc
        _client = client
    return _client


# Page through a collection, yielding the named payload fields of every
# entry. Yields nothing when the collection does not exist yet (first run).
def _iter_payloads(name, fields, page_size):
    client = _get_qdrant_client()
    if not client.collection_exists(name):
        return
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name, with_payload=fields, with_vectors=False,
            limit=page_size, offset=offset)
        for record in records:
            yield record.payload
        if offset is None:
            break


# Assemble the id + vector + payload into the PointStruct we upsert.
def _to_record(chunk, vector):
    meta = chunk["metadata"]
    entry_id = payload_id(
        meta.get("repo", ""), meta["path"],
        meta["chunk_index"], chunk["text"])
    return PointStruct(
        id=entry_id, vector=vector, payload=build_payload(chunk))


# The file extension without the dot, e.g. "map/x.vue" -> "vue".
def _file_type(path):
    return Path(path).suffix.lower().lstrip(".")



# Hex SHA-1 of a text, used in the entry id and for change detection.
def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
