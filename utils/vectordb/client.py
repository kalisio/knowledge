"""Qdrant client: create the collection, upsert points, and search it.

Talks to the Qdrant instance at QDRANT_URL, on the collection named by
QDRANT_COLLECTION. Points are built and read through points.py, so the
ingestion job (which upserts) and the API (which searches) share one payload
shape and id scheme.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import require
from utils.vectordb import points


_client = None


# Create the collection (cosine distance) if it does not exist yet.
def ensure_collection(vector_size, recreate=False):
    client = _get_client()
    name = require("QDRANT_COLLECTION")
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE),
        )


# Upsert chunks with their vectors. Same chunk -> same id, so a re-run
# overwrites its point instead of creating a duplicate.
def upsert(chunks, vectors, batch_size=64):
    client = _get_client()
    name = require("QDRANT_COLLECTION")
    records = [_to_point(c, v) for c, v in zip(chunks, vectors)]
    for start in range(0, len(records), batch_size):
        client.upsert(
            collection_name=name, points=records[start:start + batch_size])
    return len(records)


# Return the points nearest a query vector, as read_point result dicts.
def search(vector, top_k=5):
    client = _get_client()
    name = require("QDRANT_COLLECTION")
    hits = client.query_points(
        collection_name=name, query=vector, limit=top_k,
        with_payload=True).points
    return [points.read_point(hit.payload, hit.score) for hit in hits]


# Build a Qdrant point (id + vector + payload) from a chunk and its vector.
def _to_point(chunk, vector):
    meta = chunk["metadata"]
    pid = points.point_id(
        meta.get("repository", ""), meta["source_path"],
        meta["chunk_index"], chunk["text"])
    return PointStruct(
        id=pid, vector=vector, payload=points.build_payload(chunk))


# Lazily create and cache the Qdrant client.
def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=require("QDRANT_URL"))
    return _client
