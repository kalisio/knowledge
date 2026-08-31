"""What the ingestion job writes into Qdrant.

The writing half of the contract with the API: the stored payload of issue
#6 (path, repo, start_line, end_line, content), the digests that drive
change detection, and the deterministic ids. test/api/test_vectordb.py pins
the reading half, and the end-to-end suite runs the two together.
"""

import hashlib
import uuid

import ingestion.clients.vectordb as vectordb

# What issue #6 requires every stored chunk to carry. The commit history is
# part of the same contract but belongs to the file, so it is stored once per
# file and joined back on read -- see the file entry tests below.
REQUIRED_PAYLOAD_FIELDS = {
    "path", "repo", "start_line", "end_line", "content"}


# A representative production chunk: the {text, metadata{...}} shape the
# chunkers emit, after chunk_files tags it with repo and file_sha1.
def make_chunk(text="export function center () {}", repo="kdk",
               path="map/mixin.base-map.js", chunk_index=3,
               breadcrumb="baseMapMixin > center", start_line=45,
               end_line=78, commit_history=None, file_sha1="abc123"):
    metadata = {
        "path": path,
        "chunk_index": chunk_index,
        "breadcrumb": breadcrumb,
        "start_line": start_line,
        "end_line": end_line,
        "repo": repo,
        "file_sha1": file_sha1,
    }
    if commit_history is not None:
        metadata["commit_history"] = commit_history
    return {"text": text, "metadata": metadata}


# --- build_payload: what is stored in Qdrant -------------------------------

def test_the_payload_carries_every_required_field():
    payload = vectordb.build_payload(make_chunk())

    assert REQUIRED_PAYLOAD_FIELDS <= set(payload)


def test_the_payload_locates_the_chunk_in_its_file():
    payload = vectordb.build_payload(
        make_chunk(path="src/store/layers.js", repo="kano",
                   start_line=45, end_line=78))

    assert payload["path"] == "src/store/layers.js"
    assert payload["repo"] == "kano"
    assert payload["start_line"] == 45
    assert payload["end_line"] == 78


def test_the_payload_stores_the_chunk_text_as_content():
    payload = vectordb.build_payload(make_chunk(text="const a = 1"))

    assert payload["content"] == "const a = 1"


def test_the_payload_derives_the_file_type_from_the_path():
    payload = vectordb.build_payload(make_chunk(path="map/KMap.VUE"))

    assert payload["file_type"] == "vue"


def test_the_payload_defaults_the_optional_metadata():
    bare = {"text": "x", "metadata": {
        "path": "a.js", "chunk_index": 0, "start_line": 1, "end_line": 1}}

    payload = vectordb.build_payload(bare)

    assert payload["repo"] == ""
    assert payload["breadcrumb"] == ""
    assert payload["file_sha1"] == ""


def test_the_chunk_payload_does_not_carry_the_commit_history():
    # It belongs to the file: storing it here would copy it once per chunk.
    payload = vectordb.build_payload(
        make_chunk(commit_history=["fix: something"]))

    assert "commit_history" not in payload


def test_the_file_entry_id_is_deterministic_and_distinct_per_file():
    assert (vectordb.file_entry_id("kdk", "map/base.js")
            == vectordb.file_entry_id("kdk", "map/base.js"))
    assert (vectordb.file_entry_id("kdk", "map/base.js")
            != vectordb.file_entry_id("kano", "map/base.js"))
    # A file entry and a chunk of the same file are different points.
    assert (vectordb.file_entry_id("kdk", "map/base.js")
            != vectordb.payload_id("kdk", "map/base.js", 0, "text"))


# --- upsert_file_entries: the register of what has been scanned -----------

# Captures the points a call would upsert, without a Qdrant.
class _RecordingClient:
    def __init__(self):
        self.points = []

    def upsert(self, collection_name, points):
        self.points.extend(points)


def _upsert_file_entries(monkeypatch, histories, file_hashes=None):
    client = _RecordingClient()
    monkeypatch.setattr(vectordb, "_get_qdrant_client", lambda: client)
    vectordb.upsert_file_entries(histories, file_hashes)
    return {point.payload["path"]: point.payload for point in client.points}


def test_the_file_entry_carries_the_digest_of_the_scanned_file(monkeypatch):
    payloads = _upsert_file_entries(
        monkeypatch,
        {("kdk", "map/base.js"): ["fix: something"]},
        {("kdk", "map/base.js"): "abc123"})

    assert payloads["map/base.js"]["file_sha1"] == "abc123"
    assert payloads["map/base.js"]["commit_history"] == ["fix: something"]


def test_the_file_entry_records_a_file_that_yielded_no_chunk(monkeypatch):
    # The whole point of holding the digest here: a .prettierrc.json is
    # scanned but produces nothing searchable, so the code collection knows
    # nothing about it and only this entry can say it was already looked at.
    payloads = _upsert_file_entries(
        monkeypatch,
        {("kdk", ".prettierrc.json"): []},
        {("kdk", ".prettierrc.json"): "ccc"})

    assert payloads[".prettierrc.json"]["file_sha1"] == "ccc"


def test_the_file_entry_digest_is_empty_when_it_is_not_known(monkeypatch):
    payloads = _upsert_file_entries(
        monkeypatch, {("kdk", "map/base.js"): []})

    assert payloads["map/base.js"]["file_sha1"] == ""


# --- the digests -----------------------------------------------------------

def test_the_content_digest_is_the_sha1_of_the_chunk_text():
    payload = vectordb.build_payload(make_chunk(text="const a = 1"))

    assert payload["content_sha1"] == hashlib.sha1(
        b"const a = 1").hexdigest()


def test_the_content_digest_changes_with_the_text():
    first = vectordb.build_payload(make_chunk(text="const a = 1"))
    second = vectordb.build_payload(make_chunk(text="const a = 2"))

    assert first["content_sha1"] != second["content_sha1"]


def test_the_file_digest_is_carried_through_untouched():
    # It is computed once per file by chunk_files; the payload only relays it,
    # because change detection compares it across runs.
    payload = vectordb.build_payload(make_chunk(file_sha1="deadbeef"))

    assert payload["file_sha1"] == "deadbeef"


def test_every_chunk_of_a_file_shares_its_file_digest():
    payloads = [vectordb.build_payload(
        make_chunk(chunk_index=index, text=f"chunk {index}",
                   file_sha1="shared"))
        for index in range(3)]

    assert {payload["file_sha1"] for payload in payloads} == {"shared"}
    assert len({payload["content_sha1"] for payload in payloads}) == 3


# --- payload_id: deterministic and sensitive to identity + content ---------

def test_the_id_is_deterministic():
    assert (vectordb.payload_id("kdk", "map/base.js", 0, "text")
            == vectordb.payload_id("kdk", "map/base.js", 0, "text"))


def test_the_id_is_a_uuid_string():
    entry_id = vectordb.payload_id("kdk", "map/base.js", 0, "text")

    assert str(uuid.UUID(entry_id)) == entry_id


def test_the_id_changes_with_the_content():
    assert (vectordb.payload_id("kdk", "map/base.js", 0, "before")
            != vectordb.payload_id("kdk", "map/base.js", 0, "after"))


def test_the_id_distinguishes_the_repository():
    assert (vectordb.payload_id("kdk", "map/base.js", 0, "text")
            != vectordb.payload_id("kano", "map/base.js", 0, "text"))


def test_the_id_distinguishes_the_chunk_index():
    assert (vectordb.payload_id("kdk", "map/base.js", 0, "text")
            != vectordb.payload_id("kdk", "map/base.js", 1, "text"))
