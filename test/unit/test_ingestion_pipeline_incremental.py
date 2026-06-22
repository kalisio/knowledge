"""Unit tests for candidate-file filtering in ingestion.pipeline."""

import pytest


pipeline = pytest.importorskip("ingestion.pipeline")


def test_chunk_repo_respects_candidate_files(tmp_path, monkeypatch):
    repo = tmp_path / "kano"
    repo.mkdir()
    first = repo / "a.js"
    second = repo / "b.js"
    first.write_text("const a = 1\n", encoding="utf-8")
    second.write_text("const b = 2\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "scan", lambda root: [first, second])
    monkeypatch.setattr(
        pipeline, "CHUNKERS",
        {".js": lambda text, source_path: [{
            "text": text,
            "metadata": {"source_path": source_path, "chunk_index": 0},
        }]},
    )

    chunks = pipeline._chunk_repo(repo, candidate_files={"b.js"})

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source_path"] == "b.js"
    assert chunks[0]["metadata"]["repository"] == "kano"
    assert chunks[0]["metadata"]["file_sha1"]


def test_delete_existing_deduplicates_files(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        pipeline.vectordb, "delete_file",
        lambda repository, source_path: deleted.append(
            (repository, source_path)))

    pipeline._delete_existing([
        {"metadata": {"repository": "kano", "source_path": "a.js"}},
        {"metadata": {"repository": "kano", "source_path": "a.js"}},
        {"metadata": {"repository": "kano", "source_path": "b.js"}},
    ])

    assert deleted == [
        ("kano", "a.js"),
        ("kano", "b.js"),
    ]
