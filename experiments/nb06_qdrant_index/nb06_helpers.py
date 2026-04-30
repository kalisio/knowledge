"""Helpers used by nb06_qdrant_index_pipeline.ipynb."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


INDEX_VERSION = "nb06.qdrant.v1"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kalisio_qwen3_nb06_v1"
CHROMA_PATH = "vector_db/chroma_nb06"
CHROMA_COLLECTION_NAME = "kalisio_qwen3_nb06_v1"
LANCEDB_PATH = "vector_db/lancedb_nb06"
LANCEDB_TABLE_NAME = "kalisio_qwen3_nb06_v1"
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_PREFIX = (
    "Instruct: Given a developer question in French or English, retrieve "
    "the relevant Kalisio documentation page or source code file.\nQuery: "
)
QWEN_MAX_TOKENS = 8192
QWEN_GPU_BATCH_SIZE = 4


@dataclass(frozen=True)
class IndexBuildConfig:
    """Configuration for a reproducible nb06 Qdrant index build."""

    collection_name: str = COLLECTION_NAME
    qdrant_url: str = QDRANT_URL
    model_id: str = QWEN_MODEL_ID
    batch_size: int = QWEN_GPU_BATCH_SIZE
    distance: str = "cosine"
    recreate_collection: bool = True
    upsert_batch_size: int = 64


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for optional local vector-store comparisons."""

    chroma_path: str = CHROMA_PATH
    chroma_collection_name: str = CHROMA_COLLECTION_NAME
    lancedb_path: str = LANCEDB_PATH
    lancedb_table_name: str = LANCEDB_TABLE_NAME


@dataclass(frozen=True)
class GoldQuery:
    """Minimal gold-query record reused for Qdrant retrieval evaluation."""

    id: str
    layer: str
    en: str
    fr: str
    gold_sources: tuple[str, ...]
    notes: str = ""

    @property
    def is_negative(self) -> bool:
        """Return True when the query is expected to have no gold source."""
        return self.layer == "negative" or not self.gold_sources


def load_gold(path: str | Path) -> list[GoldQuery]:
    """Load the nb05 gold set without depending on nb05 helper imports."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldQuery(
            id=item["id"],
            layer=item["layer"],
            en=item["en"],
            fr=item["fr"],
            gold_sources=tuple(item.get("gold_sources", ())),
            notes=item.get("notes", ""),
        )
        for item in payload["queries"]
    ]


def build_nb06_filter_config():
    """Build the same corpus filter used by nb05, with docs and tools included."""
    from corpus_filter.models import FilterConfig
    from corpus_filter.profiles import build_js_vue_rag_profile

    base = build_js_vue_rag_profile()
    return FilterConfig(
        excluded_dirs=base.excluded_dirs - {"docs", "tools"},
        excluded_extensions=base.excluded_extensions,
        excluded_filenames=base.excluded_filenames,
        excluded_patterns=base.excluded_patterns,
        max_file_size=base.max_file_size,
        max_line_length=base.max_line_length,
        included_extensions={".md", ".js", ".mjs", ".vue", ".json"},
    )


def scan_and_chunk_corpus(root: str | Path | None = None) -> tuple[object, list[dict]]:
    """Scan the Kalisio corpus and chunk it with the current production candidates."""
    from chunking import chunk_files
    from corpus_filter import scan_corpus

    scan = scan_corpus(root=root, config=build_nb06_filter_config())
    chunks = chunk_files(scan.included)
    return scan, chunks


def sha1_text(text: str) -> str:
    """Return a stable SHA-1 hash for text content."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_point_id(source_path: str, chunk_index: int, text: str) -> str:
    """Build a deterministic UUID point id from source, chunk index, and text hash."""
    key = f"{source_path}:{chunk_index}:{sha1_text(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def source_repository(source_path: str) -> str:
    """Return the top-level repository name from a corpus-relative source path."""
    return source_path.replace("\\", "/").split("/", 1)[0]


def source_file_type(source_path: str) -> str:
    """Return the lower-case file extension without the leading dot."""
    suffix = Path(source_path).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def breadcrumb_text(metadata: dict) -> str:
    """Convert chunk breadcrumb metadata into a compact searchable string."""
    breadcrumb = metadata.get("breadcrumb", "")
    if isinstance(breadcrumb, dict):
        return " > ".join(str(v) for v in breadcrumb.values() if v)
    return str(breadcrumb or "")


def chunk_type(metadata: dict, source_path: str) -> str:
    """Infer a coarse chunk type from metadata and file extension."""
    block_type = metadata.get("block_type")
    if block_type:
        return str(block_type)
    strategy = str(metadata.get("strategy", ""))
    if "markdown" in strategy.lower() or source_path.endswith(".md"):
        return "markdown"
    if source_path.endswith(".vue"):
        return "vue"
    if source_path.endswith((".js", ".mjs")):
        return "javascript"
    if source_path.endswith(".json"):
        return "json"
    return "text"


def normalize_chunk(chunk: dict, index: int, *, index_version: str = INDEX_VERSION) -> dict:
    """Normalize one chunk into the record shape stored in Qdrant payloads."""
    text = chunk["text"]
    metadata = dict(chunk.get("metadata") or {})
    source_path = str(metadata.get("source") or "")
    chunk_index = int(metadata.get("chunk_index", index))
    text_hash = sha1_text(text)
    return {
        "id": stable_point_id(source_path, chunk_index, text),
        "text": text,
        "vector": None,
        "payload": {
            "index_version": index_version,
            "source_path": source_path,
            "repository": source_repository(source_path),
            "file_type": source_file_type(source_path),
            "chunk_type": chunk_type(metadata, source_path),
            "chunk_index": chunk_index,
            "strategy": metadata.get("strategy", ""),
            "breadcrumb": breadcrumb_text(metadata),
            "symbol": (metadata.get("breadcrumb") or {}).get("symbol", "")
            if isinstance(metadata.get("breadcrumb"), dict)
            else "",
            "text": text,
            "text_chars": len(text),
            "text_sha1": text_hash,
        },
    }


def normalize_chunks(chunks: Sequence[dict]) -> list[dict]:
    """Normalize all chunk dictionaries into stable index records."""
    return [normalize_chunk(chunk, i) for i, chunk in enumerate(chunks)]


def load_qwen_model(device: str | None = None):
    """Load the Qwen3 embedding model selected by nb05."""
    from sentence_transformers import SentenceTransformer

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model = SentenceTransformer(QWEN_MODEL_ID, device=device)
    try:
        model.max_seq_length = min(QWEN_MAX_TOKENS, model.max_seq_length)
    except AttributeError:
        pass
    return model


def encode_passages(model, texts: Sequence[str], *, batch_size: int = QWEN_GPU_BATCH_SIZE) -> np.ndarray:
    """Encode corpus passages with Qwen3 and normalized embeddings."""
    print(f"[encode] passages={len(texts)} batch_size={batch_size}")
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def encode_queries(model, queries: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
    """Encode search queries with the Qwen3 instruction prefix."""
    print(f"[encode] queries={len(queries)} batch_size={batch_size}")
    prefixed = [QWEN_QUERY_PREFIX + q for q in queries]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def attach_vectors(records: Sequence[dict], vectors: np.ndarray) -> list[dict]:
    """Attach encoded vectors to normalized records."""
    if len(records) != len(vectors):
        raise ValueError(f"records={len(records)} but vectors={len(vectors)}")
    out = []
    for record, vector in zip(records, vectors):
        item = dict(record)
        item["vector"] = vector.astype(np.float32).tolist()
        out.append(item)
    return out


def vector_records_from_chunks(
    chunks: Sequence[dict],
    model,
    *,
    batch_size: int = QWEN_GPU_BATCH_SIZE,
) -> tuple[list[dict], np.ndarray]:
    """Normalize chunks, encode them once, and attach vectors to records."""
    records = normalize_chunks(chunks)
    vectors = encode_passages(model, [record["text"] for record in records], batch_size=batch_size)
    return attach_vectors(records, vectors), vectors


def create_qdrant_client(url: str = QDRANT_URL):
    """Create a Qdrant client for the configured local service."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url)


def ensure_collection(
    client,
    *,
    collection_name: str,
    vector_size: int,
    recreate: bool = True,
) -> None:
    """Create or recreate a Qdrant collection for the nb06 dense index."""
    from qdrant_client.models import Distance, VectorParams

    if client.collection_exists(collection_name) and recreate:
        client.delete_collection(collection_name)
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def create_payload_indexes(client, collection_name: str) -> None:
    """Create payload indexes for common metadata filters when Qdrant supports them."""
    try:
        from qdrant_client.models import PayloadSchemaType
    except Exception:
        return

    for field in ("repository", "source_path", "file_type", "chunk_type", "strategy"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def iter_batches(items: Sequence[dict], batch_size: int) -> Iterable[Sequence[dict]]:
    """Yield fixed-size batches from a sequence."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def upsert_records(client, collection_name: str, records: Sequence[dict], *, batch_size: int = 64) -> int:
    """Upsert vector records into Qdrant in bounded batches."""
    from qdrant_client.models import PointStruct

    total = 0
    for batch in iter_batches(records, batch_size):
        points = [
            PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
            for item in batch
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    return total


def query_points(client, collection_name: str, query_vector: Sequence[float], *, limit: int = 10):
    """Query Qdrant and return scored points with payloads."""
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points


def build_manifest(
    *,
    config: IndexBuildConfig,
    files_included: int,
    chunks: Sequence[dict],
    vector_size: int,
    points_count: int,
    seconds: float,
) -> dict:
    """Build a JSON-serializable manifest for one index build."""
    return {
        "index_version": INDEX_VERSION,
        "collection_name": config.collection_name,
        "qdrant_url": config.qdrant_url,
        "model_id": config.model_id,
        "distance": config.distance,
        "vector_size": vector_size,
        "files_included": files_included,
        "chunks": len(chunks),
        "points_count": points_count,
        "seconds": round(seconds, 3),
        "chunks_per_sec": round(len(chunks) / max(seconds, 1e-9), 3),
    }


def write_json(path: str | Path, payload) -> None:
    """Write a JSON payload with stable UTF-8 formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_qdrant_index(
    *,
    config: IndexBuildConfig,
    chunks: Sequence[dict],
    model,
    files_included: int = 0,
    manifest_path: str | Path | None = None,
) -> dict:
    """Encode chunks, recreate the Qdrant collection, upsert points, and return a manifest."""
    started = time.perf_counter()
    records, vectors = vector_records_from_chunks(chunks, model, batch_size=config.batch_size)
    vector_size = int(vectors.shape[1])

    client = create_qdrant_client(config.qdrant_url)
    ensure_collection(
        client,
        collection_name=config.collection_name,
        vector_size=vector_size,
        recreate=config.recreate_collection,
    )
    create_payload_indexes(client, config.collection_name)
    points_count = upsert_records(
        client,
        config.collection_name,
        records,
        batch_size=config.upsert_batch_size,
    )
    elapsed = time.perf_counter() - started
    manifest = build_manifest(
        config=config,
        files_included=files_included,
        chunks=chunks,
        vector_size=vector_size,
        points_count=points_count,
        seconds=elapsed,
    )
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def build_qdrant_index_from_records(
    *,
    config: IndexBuildConfig,
    records: Sequence[dict],
    vector_size: int,
    files_included: int = 0,
    chunks: Sequence[dict] | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """Build the Qdrant collection from pre-encoded vector records."""
    started = time.perf_counter()
    client = create_qdrant_client(config.qdrant_url)
    ensure_collection(
        client,
        collection_name=config.collection_name,
        vector_size=vector_size,
        recreate=config.recreate_collection,
    )
    create_payload_indexes(client, config.collection_name)
    points_count = upsert_records(
        client,
        config.collection_name,
        records,
        batch_size=config.upsert_batch_size,
    )
    elapsed = time.perf_counter() - started
    manifest = build_manifest(
        config=config,
        files_included=files_included,
        chunks=chunks if chunks is not None else records,
        vector_size=vector_size,
        points_count=points_count,
        seconds=elapsed,
    )
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def _flat_metadata(payload: dict) -> dict:
    """Keep only Chroma/LanceDB-friendly primitive metadata values."""
    out = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


def create_chroma_client(path: str = CHROMA_PATH):
    """Create a persistent Chroma client for local vector-store comparison."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Install chromadb to run the Chroma comparison.") from exc
    return chromadb.PersistentClient(path=path)


def build_chroma_collection(
    records: Sequence[dict],
    *,
    path: str = CHROMA_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    recreate: bool = True,
    batch_size: int = 512,
):
    """Create a Chroma collection from pre-encoded vector records."""
    client = create_chroma_client(path)
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    for batch in iter_batches(list(records), batch_size):
        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=[item["vector"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[_flat_metadata(item["payload"]) for item in batch],
        )
    return collection


def chroma_file_rankings(
    collection,
    query_vectors: np.ndarray,
    *,
    limit_chunks: int = 500,
    max_files: int = 20,
) -> list[list[str]]:
    """Query Chroma vectors and return de-duplicated file-level rankings."""
    rankings: list[list[str]] = []
    for vector in query_vectors:
        result = collection.query(
            query_embeddings=[vector.astype(np.float32).tolist()],
            n_results=limit_chunks,
            include=["metadatas"],
        )
        seen: set[str] = set()
        ranking: list[str] = []
        for metadata in result.get("metadatas", [[]])[0]:
            source_path = str((metadata or {}).get("source_path") or "")
            if not source_path or source_path in seen:
                continue
            seen.add(source_path)
            ranking.append(source_path)
            if len(ranking) >= max_files:
                break
        rankings.append(ranking)
    return rankings


def create_lancedb_connection(path: str = LANCEDB_PATH):
    """Create a LanceDB connection for local vector-store comparison."""
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError("Install lancedb to run the LanceDB comparison.") from exc
    Path(path).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(path)


def build_lancedb_table(
    records: Sequence[dict],
    *,
    path: str = LANCEDB_PATH,
    table_name: str = LANCEDB_TABLE_NAME,
    mode: str = "overwrite",
):
    """Create a LanceDB table from pre-encoded vector records."""
    db = create_lancedb_connection(path)
    rows = []
    for item in records:
        payload = _flat_metadata(item["payload"])
        rows.append({
            "id": item["id"],
            "vector": item["vector"],
            "text": item["text"],
            **payload,
        })
    return db.create_table(table_name, data=rows, mode=mode)


def lancedb_file_rankings(
    table,
    query_vectors: np.ndarray,
    *,
    limit_chunks: int = 500,
    max_files: int = 20,
) -> list[list[str]]:
    """Query LanceDB vectors and return de-duplicated file-level rankings."""
    rankings: list[list[str]] = []
    for vector in query_vectors:
        rows = table.search(vector.astype(np.float32).tolist()).limit(limit_chunks).to_list()
        seen: set[str] = set()
        ranking: list[str] = []
        for row in rows:
            source_path = str(row.get("source_path") or "")
            if not source_path or source_path in seen:
                continue
            seen.add(source_path)
            ranking.append(source_path)
            if len(ranking) >= max_files:
                break
        rankings.append(ranking)
    return rankings


def file_rank_from_points(points, *, max_files: int = 20) -> list[str]:
    """Convert Qdrant chunk hits into a de-duplicated file-level ranking."""
    seen: set[str] = set()
    out: list[str] = []
    for point in points:
        payload = point.payload or {}
        source_path = str(payload.get("source_path") or payload.get("source") or "")
        if not source_path or source_path in seen:
            continue
        seen.add(source_path)
        out.append(source_path)
        if len(out) >= max_files:
            break
    return out


def qdrant_file_rankings(
    client,
    collection_name: str,
    query_vectors: np.ndarray,
    *,
    limit_chunks: int = 50,
    max_files: int = 20,
) -> list[list[str]]:
    """Query Qdrant vectors and return file-level rankings for each query."""
    rankings: list[list[str]] = []
    for vector in query_vectors:
        points = query_points(
            client,
            collection_name,
            vector.astype(np.float32).tolist(),
            limit=limit_chunks,
        )
        rankings.append(file_rank_from_points(points, max_files=max_files))
    return rankings


def hit_at(retrieved: Sequence[str], gold: Sequence[str]) -> int:
    """Return 1 when any gold file appears in the retrieved file list."""
    if not gold:
        return 0
    gold_set = set(gold)
    return int(any(path in gold_set for path in retrieved))


def recall_at(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Compute file-level recall for a retrieved file list."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    return sum(1 for path in retrieved if path in gold_set) / len(gold_set)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Compute reciprocal rank of the first retrieved gold file."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    for idx, path in enumerate(retrieved, start=1):
        if path in gold_set:
            return 1.0 / idx
    return 0.0


def evaluate_file_rankings(
    rankings: Sequence[Sequence[str]],
    queries: Sequence,
    *,
    language: str,
    approach: str,
    k: int = 5,
    ks: tuple[int, ...] = (1, 5, 10),
) -> list[dict]:
    """Evaluate Qdrant file-level rankings against nb05 gold queries."""
    if len(rankings) != len(queries):
        raise ValueError(f"rankings={len(rankings)} but queries={len(queries)}")

    positive_gold: set[str] = set()
    for query in queries:
        if not query.is_negative:
            positive_gold.update(query.gold_sources)

    metric_ks = tuple(sorted(set(ks) | {k}))
    rows: list[dict] = []
    for ranking, query in zip(rankings, queries):
        top_k = list(ranking[:k])
        metrics: dict[str, float | int] = {}
        for metric_k in metric_ks:
            current_top = list(ranking[:metric_k])
            if query.is_negative:
                metrics[f"hit@{metric_k}"] = int(
                    not any(path in positive_gold for path in current_top)
                )
                metrics[f"recall@{metric_k}"] = float("nan")
            else:
                metrics[f"hit@{metric_k}"] = hit_at(current_top, query.gold_sources)
                metrics[f"recall@{metric_k}"] = recall_at(current_top, query.gold_sources)
        if query.is_negative:
            hit = int(not any(path in positive_gold for path in top_k))
            recall = float("nan")
            rr = float("nan")
        else:
            hit = hit_at(top_k, query.gold_sources)
            recall = recall_at(top_k, query.gold_sources)
            rr = reciprocal_rank(ranking, query.gold_sources)

        rows.append({
            "approach": approach,
            "language": language,
            "layer": query.layer,
            "query_id": query.id,
            "is_negative": query.is_negative,
            "hit@k": hit,
            "recall@k": recall,
            "mrr": rr,
            **metrics,
        })
    return rows
