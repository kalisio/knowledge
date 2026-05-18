"""Entry point for rebuilding the Kalisio Qdrant index.

Driven entirely by environment variables — see `IngestionConfig` in
`rag_system/config.py`. No CLI flags: configuration lives in
`development/workspaces/services/knowledge/knowledge.dec.env` and is sourced
by `services.sh knowledge` before invocation, matching the convention used
by every other Kalisio service.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from .chunking.api import chunk_files
from .corpus_filter.api import scan_corpus
from .corpus_filter.models import FilterConfig, FileRecord
from .corpus_filter.profiles import build_default_profile
from .rag_system.config import IngestionConfig


def build_ingestion_filter_config() -> FilterConfig:
    """Filter config for scanning live Kalisio repositories.

    Keeps `docs/` in scope so each repo's documentation markdown is indexed,
    and restricts inputs to the four extensions the RAG pipeline supports.
    """
    base = build_default_profile()
    return replace(
        base,
        included_extensions={".md", ".js", ".mjs", ".vue", ".json"},
    )


def profile_records(records: Iterable[FileRecord], profile: str) -> list[FileRecord]:
    prefix = f"{profile}/"
    return [record for record in records if record.rel_path.replace("\\", "/").startswith(prefix)]


def source_path(record: FileRecord) -> str:
    return record.rel_path.replace("\\", "/")


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_sha1_map(records: Sequence[FileRecord]) -> dict[str, str]:
    return {source_path(record): sha1_file(record.path) for record in records}


def changed_records(
    records: Sequence[FileRecord],
    file_sha1s: dict[str, str],
    indexed_manifest: dict[str, str],
) -> list[FileRecord]:
    return [
        record
        for record in records
        if indexed_manifest.get(source_path(record)) != file_sha1s[source_path(record)]
    ]


def stale_source_paths(records: Sequence[FileRecord], indexed_manifest: dict[str, str]) -> list[str]:
    alive = {source_path(record) for record in records}
    return sorted(set(indexed_manifest) - alive)


def chunk_records(records: Sequence[FileRecord], file_sha1s: dict[str, str]) -> list[dict]:
    chunks = chunk_files(records)
    for chunk in chunks:
        metadata = chunk.setdefault("metadata", {})
        src = str(metadata.get("source") or "")
        metadata["file_sha1"] = file_sha1s.get(src, "")
    return chunks


def encode_chunks(model, chunks: Sequence[dict], batch_size: int):
    return model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embedding_runtime(cfg: IngestionConfig, state: dict):
    if "model" not in state:
        from .embedding_utils import embed_batch_size, load_embedding_model

        state["model"] = load_embedding_model(cfg.embedding_model)
        state["batch_size"] = cfg.embedding_batch_size or embed_batch_size()
    return state["model"], int(state["batch_size"])


def run_profile(
    *,
    cfg: IngestionConfig,
    scan_records: Sequence[FileRecord],
    profile: str,
    recreate: bool,
    embedding_state: dict,
) -> tuple[int, int]:
    records = profile_records(scan_records, profile)
    file_sha1s = build_file_sha1_map(records)
    print(f"[scan] root={cfg.ingestion_root} profile={profile} files={len(records)}")

    client = None
    indexed_manifest: dict[str, str] = {}
    if cfg.incremental or not cfg.dry_run:
        from .rag_system.qdrant_store import create_client, scroll_source_manifest

        client = create_client(cfg.qdrant_url)
        if cfg.incremental and not recreate:
            indexed_manifest = scroll_source_manifest(client, cfg.qdrant_collection, profile=profile)

    target_records = changed_records(records, file_sha1s, indexed_manifest) if cfg.incremental else records
    stale_paths = stale_source_paths(records, indexed_manifest) if cfg.incremental else []
    print(f"[change] profile={profile} changed_files={len(target_records)} stale_source_paths={len(stale_paths)}")

    chunks = chunk_records(target_records, file_sha1s)
    chunked_sources = {str((chunk.get("metadata") or {}).get("source") or "") for chunk in chunks}
    zero_chunk_files: list[tuple[str, str]] = [
        (source_path(record), file_sha1s[source_path(record)])
        for record in target_records
        if source_path(record) not in chunked_sources
    ]
    print(f"[chunk] profile={profile} chunks={len(chunks)} zero_chunk_files={len(zero_chunk_files)}")

    if cfg.dry_run:
        return len(chunks), 0

    if client is None:
        from .rag_system.qdrant_store import create_client

        client = create_client(cfg.qdrant_url)

    from .rag_system.qdrant_store import (
        collection_vector_size,
        delete_source_path,
        ensure_collection,
        prune_missing_source_paths,
        upsert_chunks,
        upsert_sentinels,
    )

    if not chunks:
        if not cfg.incremental and not target_records:
            raise SystemExit("No chunks produced; refusing to create an empty index.")
        if not cfg.no_prune:
            for stale_path in stale_paths:
                delete_source_path(client, cfg.qdrant_collection, stale_path, profile=profile)
            print(f"[prune] profile={profile} stale_source_paths={len(stale_paths)}")
        sentinels_written = 0
        if zero_chunk_files and client.collection_exists(cfg.qdrant_collection):
            vsize = collection_vector_size(client, cfg.qdrant_collection)
            for src, _ in zero_chunk_files:
                delete_source_path(client, cfg.qdrant_collection, src, profile=profile)
            sentinels_written = upsert_sentinels(
                client,
                cfg.qdrant_collection,
                zero_chunk_files,
                vector_size=vsize,
                index_version=cfg.index_version,
                profile=profile,
            )
        print(f"[sentinel] profile={profile} written={sentinels_written}")
        print(f"[upsert] profile={profile} points=0 collection={cfg.qdrant_collection} qdrant={cfg.qdrant_url}")
        return 0, 0

    model, batch_size = embedding_runtime(cfg, embedding_state)
    vectors = encode_chunks(model, chunks, batch_size)

    ensure_collection(client, cfg.qdrant_collection, vector_size=vectors.shape[1], recreate=recreate)

    replaced_source_paths = sorted({source_path(record) for record in target_records})
    for replaced_source_path in replaced_source_paths:
        delete_source_path(client, cfg.qdrant_collection, replaced_source_path, profile=profile)
    print(f"[cleanup] profile={profile} replaced_source_paths={len(replaced_source_paths)}")

    upserted = upsert_chunks(
        client,
        cfg.qdrant_collection,
        chunks,
        vectors,
        batch_size=batch_size,
        index_version=cfg.index_version,
        profile=profile,
    )
    print(f"[upsert] profile={profile} points={upserted} collection={cfg.qdrant_collection} qdrant={cfg.qdrant_url}")

    sentinels_written = upsert_sentinels(
        client,
        cfg.qdrant_collection,
        zero_chunk_files,
        vector_size=vectors.shape[1],
        index_version=cfg.index_version,
        profile=profile,
    )
    print(f"[sentinel] profile={profile} written={sentinels_written}")

    if not cfg.no_prune:
        alive_source_paths = set(file_sha1s)
        pruned = prune_missing_source_paths(client, cfg.qdrant_collection, alive_source_paths, profile=profile)
        print(f"[prune] profile={profile} stale_source_paths={pruned}")

    return len(chunks), upserted


def main() -> None:
    cfg = IngestionConfig()
    root = cfg.ingestion_root.resolve()

    missing = [name for name in cfg.repos if not (root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"INGESTION_REPOS lists repositories missing under {root}: {', '.join(missing)}. "
            "Run `k-clone kalisio all` to fetch them."
        )

    scan = scan_corpus(root=root, config=build_ingestion_filter_config(), profile="ingestion")
    print(f"[scan] root={root} repos={','.join(cfg.repos)} excluded={len(scan.excluded)}")

    total_chunks = 0
    total_upserted = 0
    embedding_state: dict = {}
    for index, profile in enumerate(cfg.repos):
        chunks, upserted = run_profile(
            cfg=cfg,
            scan_records=scan.included,
            profile=profile,
            recreate=cfg.recreate and index == 0,
            embedding_state=embedding_state,
        )
        total_chunks += chunks
        total_upserted += upserted
    print(f"[done] repos={len(cfg.repos)} chunks={total_chunks} upserted={total_upserted}")


if __name__ == "__main__":
    main()
