"""Qdrant storage helpers for the v1 RAG system."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_point_id(source_path: str, chunk_index: int, text: str) -> str:
    key = f"{source_path}:{chunk_index}:{sha1_text(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def source_repository(source_path: str) -> str:
    return source_path.replace("\\", "/").split("/", 1)[0]


def source_file_type(source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def breadcrumb_text(metadata: dict[str, Any]) -> str:
    breadcrumb = metadata.get("breadcrumb", "")
    if isinstance(breadcrumb, dict):
        return " > ".join(str(value) for value in breadcrumb.values() if value)
    return str(breadcrumb or "")


def chunk_type(metadata: dict[str, Any], source_path: str) -> str:
    block_type = metadata.get("block_type")
    if block_type:
        return str(block_type)
    if source_path.endswith(".vue"):
        return "vue"
    if source_path.endswith((".js", ".mjs")):
        return "javascript"
    if source_path.endswith(".json"):
        return "json"
    if source_path.endswith(".md"):
        return "markdown"
    return "text"


def normalize_chunk(
    chunk: dict[str, Any],
    index: int,
    *,
    source_path: str | None = None,
    file_sha1: str = "",
    index_version: str = "system.v1",
) -> dict[str, Any]:
    text = chunk["text"]
    metadata = dict(chunk.get("metadata") or {})
    resolved_source = str(source_path or metadata.get("source") or "")
    chunk_index = int(metadata.get("chunk_index", index))
    text_hash = sha1_text(text)
    breadcrumb = metadata.get("breadcrumb")
    symbol = breadcrumb.get("symbol", "") if isinstance(breadcrumb, dict) else ""
    return {
        "id": stable_point_id(resolved_source, chunk_index, text),
        "text": text,
        "vector": None,
        "payload": {
            "index_version": index_version,
            "source_path": resolved_source,
            "repository": source_repository(resolved_source),
            "file_type": source_file_type(resolved_source),
            "chunk_type": chunk_type(metadata, resolved_source),
            "chunk_index": chunk_index,
            "strategy": metadata.get("strategy", ""),
            "breadcrumb": breadcrumb_text(metadata),
            "symbol": symbol,
            "text": text,
            "text_chars": len(text),
            "text_sha1": text_hash,
            "file_sha1": file_sha1,
        },
    }


def normalize_chunks(
    chunks: Sequence[dict[str, Any]],
    *,
    source_path: str | None = None,
    file_sha1: str = "",
    index_version: str = "system.v1",
) -> list[dict[str, Any]]:
    return [
        normalize_chunk(
            chunk,
            index,
            source_path=source_path,
            file_sha1=file_sha1,
            index_version=index_version,
        )
        for index, chunk in enumerate(chunks)
    ]


def attach_vectors(records: Sequence[dict[str, Any]], vectors: np.ndarray) -> list[dict[str, Any]]:
    if len(records) != len(vectors):
        raise ValueError(f"records={len(records)} but vectors={len(vectors)}")
    out: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        item = dict(record)
        item["vector"] = vector.astype(np.float32).tolist()
        out.append(item)
    return out


def create_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    vector_size: int,
    recreate: bool = False,
) -> None:
    if client.collection_exists(collection_name) and recreate:
        client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    for field in ("repository", "source_path", "file_type", "chunk_type", "strategy", "index_version"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def iter_batches(items: Sequence[dict[str, Any]], batch_size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def upsert_records(
    client: QdrantClient,
    collection_name: str,
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int = 64,
) -> int:
    total = 0
    for batch in iter_batches(list(records), batch_size):
        points = [
            PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
            for item in batch
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    return total


def delete_source_path(client: QdrantClient, collection_name: str, source_path: str) -> None:
    if not client.collection_exists(collection_name):
        return
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="source_path",
                        match=MatchValue(value=source_path),
                    )
                ]
            )
        ),
    )


def query_points(
    client: QdrantClient,
    collection_name: str,
    query_vector: Sequence[float],
    *,
    limit: int,
):
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points

