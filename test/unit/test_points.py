"""Unit tests for the Qdrant point contract (utils/vectordb/points.py).

points.py is the shared payload schema and id scheme that the ingestion job
(writer) and the API (reader) both rely on, so these tests pin its
production contract: deterministic ids, the full stored payload, and a
write/read round-trip that exposes only the search-facing fields.
"""

import hashlib
import uuid

from utils.vectordb import points


# A representative production chunk: the {text, metadata{...}} shape the
# chunkers emit, after the pipeline tags it with repository and file_sha1.
def make_chunk(text="def center(): pass", repository="kdk",
               source_path="map/mixin.base-map.js", chunk_index=3,
               breadcrumb="baseMapMixin > center",
               commit_history=None, file_sha1="abc123"):
    metadata = {
        "source_path": source_path,
        "chunk_index": chunk_index,
        "breadcrumb": breadcrumb,
        "repository": repository,
        "file_sha1": file_sha1,
    }
    if commit_history is not None:
        metadata["commit_history"] = commit_history
    return {"text": text, "metadata": metadata}


# --- point_id: deterministic and sensitive to identity + content ----------

def test_point_id_is_deterministic():
    args = ("kdk", "map/x.js", 3, "some code")
    assert points.point_id(*args) == points.point_id(*args)


def test_point_id_is_a_uuid_string():
    pid = points.point_id("kdk", "map/x.js", 3, "some code")
    assert str(uuid.UUID(pid)) == pid


def test_point_id_changes_with_content():
    base = ("kdk", "map/x.js", 3)
    assert points.point_id(*base, "v1") != points.point_id(*base, "v2")


def test_point_id_distinguishes_repository():
    # source_path is repo-relative, so the same path in two repos must not
    # collide on the same id.
    assert (points.point_id("kdk", "x.js", 0, "t")
            != points.point_id("kano", "x.js", 0, "t"))


def test_point_id_distinguishes_chunk_index():
    assert (points.point_id("kdk", "x.js", 0, "t")
            != points.point_id("kdk", "x.js", 1, "t"))


# --- build_payload: full stored contract ----------------------------------

def test_build_payload_has_the_full_contract():
    payload = points.build_payload(make_chunk())
    assert set(payload) == {
        "text", "source_path", "repository", "file_type", "chunk_index",
        "breadcrumb", "commit_history", "text_sha1", "file_sha1",
    }


def test_build_payload_derives_file_type_and_text_sha1():
    payload = points.build_payload(
        make_chunk(text="hello", source_path="docs/readme.MD"))
    assert payload["file_type"] == "md"  # lowercased, dot stripped
    assert payload["text_sha1"] == hashlib.sha1(b"hello").hexdigest()


def test_build_payload_defaults_for_missing_optional_metadata():
    # Only the required keys present; optionals must fall back, not raise.
    chunk = {"text": "x", "metadata": {
        "source_path": "a.js", "chunk_index": 0}}
    payload = points.build_payload(chunk)
    assert payload["repository"] == ""
    assert payload["breadcrumb"] == ""
    assert payload["commit_history"] == []
    assert payload["file_sha1"] == ""


# --- read_point: search-facing view, write/read round-trip ----------------

def test_read_point_round_trip_keeps_search_facing_fields():
    chunk = make_chunk(commit_history=["fix bug"])
    result = points.read_point(points.build_payload(chunk), score=0.87)

    assert result["source_path"] == chunk["metadata"]["source_path"]
    assert result["repository"] == chunk["metadata"]["repository"]
    assert result["breadcrumb"] == chunk["metadata"]["breadcrumb"]
    assert result["chunk_index"] == chunk["metadata"]["chunk_index"]
    assert result["commit_history"] == ["fix bug"]
    assert result["content"] == chunk["text"]  # text is exposed as content
    assert result["score"] == 0.87


def test_read_point_exposes_only_search_facing_fields():
    # Storage-internal fields (text_sha1, file_sha1, file_type) are not part
    # of a search result.
    result = points.read_point(points.build_payload(make_chunk()), score=0.5)
    assert set(result) == {
        "source_path", "repository", "breadcrumb", "chunk_index",
        "content", "commit_history", "score",
    }
