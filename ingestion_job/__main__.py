"""Command-line entry point for rebuilding the Kalisio Qdrant index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from .chunking.api import chunk_files
from .corpus_filter.api import DEFAULT_SCAN_ROOT, scan_corpus
from .corpus_filter.models import FileRecord, FilterConfig
from .corpus_filter.profiles import build_js_vue_rag_profile
from .rag_system.config import RuntimeConfig


DEFAULT_CORPUS_PROFILE = "kdk"
CORPUS_PROFILES = ("crisis", "kano", "kapp", "kdk", "skeleton")


def build_ingestion_filter_config() -> FilterConfig:
    """Return the nb06 production-oriented corpus filter."""
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


def corpus_profile_from_env() -> str:
    return os.getenv("INGESTION_PROFILE", DEFAULT_CORPUS_PROFILE)


def profile_records(records: Iterable[FileRecord], profile: str) -> list[FileRecord]:
    prefix = f"{profile}/"
    return [record for record in records if record.rel_path.replace("\\", "/").startswith(prefix)]


def records_for_profile(scan_records: Iterable[FileRecord], root: Path, profile: str) -> list[FileRecord]:
    """Select a corpus profile while keeping source paths repository-prefixed."""
    records = list(scan_records)
    selected = profile_records(records, profile)
    if selected or root.name != profile:
        return selected

    prefixed: list[FileRecord] = []
    for record in records:
        rel_path = record.rel_path.replace("\\", "/")
        prefixed.append(
            FileRecord(
                path=record.path,
                rel_path=f"{profile}/{rel_path}",
                extension=record.extension,
                size=record.size,
                zone=profile,
                exclude_reason=record.exclude_reason,
            )
        )
    return prefixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan, chunk, embed, and upsert Kalisio corpus chunks.")
    parser.add_argument(
        "--profile",
        default=corpus_profile_from_env(),
        help="Corpus profile under data/ to ingest, for example kdk or kano.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name. Defaults to QDRANT_COLLECTION or RuntimeConfig default.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help="Corpus root. Defaults to knowledge/data so source paths keep the repo prefix.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding/upsert batch size. Defaults to EMBEDDING_BATCH_SIZE or device-aware helper.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Stop after scan and chunk; do not embed or upsert.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the target collection first.")
    parser.add_argument("--no-prune", action="store_true", help="Do not delete stale chunks for the selected profile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig()
    collection = args.collection or cfg.qdrant_collection
    root = args.root.resolve()

    if args.profile not in CORPUS_PROFILES:
        known = ", ".join(CORPUS_PROFILES)
        raise SystemExit(f"Unknown ingestion profile {args.profile!r}. Known profiles: {known}")

    scan = scan_corpus(root=root, config=build_ingestion_filter_config(), profile="ingestion")
    records = records_for_profile(scan.included, root, args.profile)
    print(f"[scan] root={root} profile={args.profile} files={len(records)} excluded={len(scan.excluded)}")

    chunks = chunk_files(records)
    print(f"[chunk] chunks={len(chunks)}")

    if args.dry_run:
        return

    if not chunks:
        raise SystemExit("No chunks produced; refusing to create an empty index.")

    from .embedding_utils import embed_batch_size, load_embedding_model
    from .rag_system.qdrant_store import (
        create_client,
        ensure_collection,
        prune_missing_source_paths,
        upsert_chunks,
    )

    model = load_embedding_model(cfg.embedding_model)
    batch_size = args.batch_size or cfg.embedding_batch_size or embed_batch_size()
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    client = create_client(cfg.qdrant_url)
    ensure_collection(client, collection, vector_size=vectors.shape[1], recreate=args.recreate)
    upserted = upsert_chunks(
        client,
        collection,
        chunks,
        vectors,
        batch_size=batch_size,
        index_version=cfg.index_version,
        profile=args.profile,
    )
    print(f"[upsert] points={upserted} collection={collection} qdrant={cfg.qdrant_url}")

    if not args.no_prune:
        alive_source_paths = {record.rel_path.replace("\\", "/") for record in records}
        pruned = prune_missing_source_paths(client, collection, alive_source_paths, profile=args.profile)
        print(f"[prune] stale_source_paths={pruned}")


if __name__ == "__main__":
    main()
