from pathlib import Path

from ingestion_job.__main__ import build_file_sha1_map, changed_records, stale_source_paths
from ingestion_job.corpus_filter.models import FileRecord
from ingestion_job.rag_system.qdrant_store import normalize_chunk


def _record(path: Path, rel_path: str) -> FileRecord:
    return FileRecord(
        path=path,
        rel_path=rel_path,
        extension=path.suffix,
        size=path.stat().st_size,
        zone=rel_path.split("/", 1)[0],
    )


def test_changed_records_detect_new_changed_and_stale_files(tmp_path: Path) -> None:
    unchanged = tmp_path / "unchanged.md"
    changed = tmp_path / "changed.md"
    new = tmp_path / "new.md"
    unchanged.write_text("same", encoding="utf-8")
    changed.write_text("new content", encoding="utf-8")
    new.write_text("brand new", encoding="utf-8")

    records = [
        _record(unchanged, "kdk/docs/unchanged.md"),
        _record(changed, "kdk/docs/changed.md"),
        _record(new, "kdk/docs/new.md"),
    ]
    file_sha1s = build_file_sha1_map(records)
    indexed_manifest = {
        "kdk/docs/unchanged.md": file_sha1s["kdk/docs/unchanged.md"],
        "kdk/docs/changed.md": "old-hash",
        "kdk/docs/deleted.md": "deleted-hash",
    }

    changed_paths = {record.rel_path for record in changed_records(records, file_sha1s, indexed_manifest)}

    assert changed_paths == {"kdk/docs/changed.md", "kdk/docs/new.md"}
    assert stale_source_paths(records, indexed_manifest) == ["kdk/docs/deleted.md"]


def test_normalize_chunk_prefers_metadata_file_sha1() -> None:
    record = normalize_chunk(
        {
            "text": "Context: Demo\n\nbody",
            "metadata": {
                "source": "kdk/docs/demo.md",
                "chunk_index": 0,
                "strategy": "D_ast_breadcrumb",
                "file_sha1": "abc123",
            },
        },
        99,
        profile="kdk",
    )

    assert record["payload"]["profile"] == "kdk"
    assert record["payload"]["file_sha1"] == "abc123"
