"""Reads the Kalisio corpus out of Qdrant."""

import uuid

from qdrant_client import QdrantClient

from api.config import get_config
from api.logger import get_logger

_client = None

log = get_logger("vectordb")


# Raised when Qdrant cannot be reached at all. The API turns it into a 503:
# a dependency being down is not a bug in the request.
class QdrantUnreachable(RuntimeError):
    pass


# Return the entries nearest a query vector, best match first. Empty list if
# the collection does not exist yet (corpus never ingested), so callers can
# surface a "run ingestion" hint instead of a 500.
def search(vector, top_k=5):
    client = _get_client()
    name = get_config().qdrant_collection_code
    if not client.collection_exists(name):
        return []
    hits = client.query_points(
        collection_name=name, query=vector, limit=top_k,
        with_payload=True).points
    results = [read_payload(hit.payload, hit.score) for hit in hits]
    # One extra round trip fills in the history of every file quoted, which
    # is stored once per file rather than on each of its chunks.
    histories = get_commit_histories(
        {(result["repo"], result["path"]) for result in results})
    for result in results:
        result["commit_history"] = histories.get(
            (result["repo"], result["path"]), [])
    return results


# The number of indexed chunks, or 0 when the collection does not exist yet
# ("index not built" vs "query had no match").
def count_chunks():
    client = _get_client()
    name = get_config().qdrant_collection_code
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name).count


# The commit history of each (repo, path) asked for, in one round trip.
# Missing files simply do not appear in the result.
def get_commit_histories(file_keys):
    file_keys = list(file_keys)
    if not file_keys:
        return {}
    client = _get_client()
    name = get_config().qdrant_collection_files
    if not client.collection_exists(name):
        return {}
    records = client.retrieve(
        collection_name=name,
        ids=[file_entry_id(repo, path) for repo, path in file_keys],
        with_payload=True, with_vectors=False)
    return {(record.payload["repo"], record.payload["path"]):
            record.payload.get("commit_history", [])
            for record in records}


# Turn a stored payload and its similarity score into a search result. This
# is the shape callers see, and the reading half of the contract with the
# ingestion job: the line range is rendered as "45-78", the way an editor
# takes it. commit_history starts empty and is filled in by search().
def read_payload(payload, score):
    return {
        "path": payload.get("path", ""),
        "repo": payload.get("repo", ""),
        "lines": _line_range(payload),
        "score": score,
        "content": payload.get("content", ""),
        "commit_history": [],
        "breadcrumb": payload.get("breadcrumb", ""),
        "chunk_index": payload.get("chunk_index", 0),
    }


# Deterministic id of a file entry, the same one the ingestion job writes.
def file_entry_id(repository, path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"file:{repository}:{path}"))


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Create & cache the Qdrant client. An unreachable Qdrant raises rather than
# exiting: the API loses its index, not its ability to answer /health.
def _get_client():
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


# Render a chunk's line range the way an editor takes it: "45-78", or "45"
# when the chunk sits on a single line.
def _line_range(payload):
    start = payload.get("start_line")
    end = payload.get("end_line")
    if start is None or end is None:
        return ""
    return str(start) if start == end else f"{start}-{end}"
