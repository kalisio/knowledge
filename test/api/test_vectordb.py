"""What the API reads back out of Qdrant.

The reading half of the contract with the ingestion job: the shape /search
promises, and the line range rendered the way an editor takes it.
test/ingestion/test_vectordb.py pins the writing half, and the end-to-end
suite runs the two together.
"""

import api.services.vectordb as vectordb


# A stored payload, as the ingestion job writes it.
def make_payload(path="src/store/layers.js", repo="kano", start_line=45,
                 end_line=78, content="export const base = []",
                 breadcrumb="layers > base", chunk_index=3):
    return {
        "path": path,
        "repo": repo,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        "breadcrumb": breadcrumb,
        "chunk_index": chunk_index,
        "file_type": "js",
        "content_sha1": "abc123",
        "file_sha1": "def456",
    }


# --- read_payload: the shape /search returns -------------------------------

def test_the_result_has_the_shape_the_api_promises():
    result = vectordb.read_payload(make_payload(), score=0.92)

    assert {"path", "lines", "score", "content", "commit_history"} <= set(result)


def test_the_result_carries_what_was_stored():
    result = vectordb.read_payload(make_payload(), score=0.92)

    assert result["path"] == "src/store/layers.js"
    assert result["repo"] == "kano"
    assert result["score"] == 0.92
    assert result["content"] == "export const base = []"
    assert result["breadcrumb"] == "layers > base"
    assert result["chunk_index"] == 3


def test_the_result_renders_the_line_range_for_an_editor():
    result = vectordb.read_payload(
        make_payload(start_line=45, end_line=78), 0.9)

    assert result["lines"] == "45-78"


def test_a_single_line_chunk_reports_one_line():
    result = vectordb.read_payload(
        make_payload(start_line=12, end_line=12), 0.9)

    assert result["lines"] == "12"


def test_a_payload_without_a_range_reports_no_lines():
    result = vectordb.read_payload({"path": "a.js", "content": "x"}, 0.5)

    assert result["lines"] == ""


def test_the_commit_history_starts_empty():
    # It is stored once per file and joined in by search(), not carried by
    # the chunk payload.
    assert vectordb.read_payload(make_payload(), 0.5)["commit_history"] == []


def test_the_result_hides_the_bookkeeping_fields():
    # The digests and the raw line numbers drive ingestion, not retrieval.
    result = vectordb.read_payload(make_payload(), score=0.5)

    assert "file_sha1" not in result
    assert "content_sha1" not in result
    assert "start_line" not in result


def test_a_missing_payload_field_falls_back_to_an_empty_value():
    result = vectordb.read_payload({}, score=0.1)

    assert result["path"] == ""
    assert result["repo"] == ""
    assert result["content"] == ""
    assert result["breadcrumb"] == ""


# --- file entries: both sides must agree on the id -------------------------

def test_the_file_entry_id_is_deterministic():
    assert (vectordb.file_entry_id("kdk", "map/base.js")
            == vectordb.file_entry_id("kdk", "map/base.js"))


def test_the_file_entry_id_distinguishes_the_repository():
    assert (vectordb.file_entry_id("kdk", "map/base.js")
            != vectordb.file_entry_id("kano", "map/base.js"))
