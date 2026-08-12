from ingestion import indexed_file_state


# --- compute_file_sha1: stable whole-file digest --------------------------

def test_compute_file_sha1_is_the_hex_sha1_of_the_utf8_bytes():
    # Pinned digest: it is stored in Qdrant and compared against the next
    # run, so changing the algorithm or the encoding would mark every
    # indexed file as changed and re-embed the whole corpus. The accented
    # character makes the utf-8 choice observable.
    assert (indexed_file_state.compute_file_sha1("héllo")
            == "35b5ea45c5e41f78b46a937cc74d41dfea920890")


def test_compute_file_sha1_is_deterministic():
    assert (indexed_file_state.compute_file_sha1("const a = 1")
            == indexed_file_state.compute_file_sha1("const a = 1"))


def test_compute_file_sha1_changes_with_content():
    assert (indexed_file_state.compute_file_sha1("const a = 1")
            != indexed_file_state.compute_file_sha1("const a = 2"))


# --- load_indexed_file_hashes: rebuild the state from stored payloads -----

def test_load_indexed_file_hashes_keys_on_repository_and_source_path(
        monkeypatch):
    monkeypatch.setattr(indexed_file_state.vectordb, "iter_payloads", lambda: [
        {"repository": "kdk", "source_path": "map/x.js", "file_sha1": "aaa"},
        {"repository": "kano", "source_path": "map/x.js", "file_sha1": "bbb"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {
        ("kdk", "map/x.js"): "aaa",
        ("kano", "map/x.js"): "bbb",
    }


def test_load_indexed_file_hashes_shares_one_entry_per_file(monkeypatch):
    # Every chunk of a file carries the same file_sha1, so many payloads
    # collapse into a single entry.
    monkeypatch.setattr(indexed_file_state.vectordb, "iter_payloads", lambda: [
        {"repository": "kdk", "source_path": "map/x.js", "file_sha1": "aaa"},
        {"repository": "kdk", "source_path": "map/x.js", "file_sha1": "aaa"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {
        ("kdk", "map/x.js"): "aaa"}


def test_load_indexed_file_hashes_skips_payloads_without_a_digest(monkeypatch):
    monkeypatch.setattr(indexed_file_state.vectordb, "iter_payloads", lambda: [
        {"repository": "kdk", "source_path": "map/x.js", "file_sha1": ""},
        {"repository": "kdk", "source_path": "map/y.js"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {}


def test_load_indexed_file_hashes_is_empty_on_the_first_run(monkeypatch):
    monkeypatch.setattr(
        indexed_file_state.vectordb, "iter_payloads", lambda: [])
    assert indexed_file_state.load_indexed_file_hashes() == {}


# --- select_changed_chunks: drop chunks whose file is unchanged -----------

def test_select_changed_chunks_keeps_a_file_that_is_not_indexed_yet():
    chunk = {"metadata": {
        "repository": "kdk", "source_path": "map/x.js", "file_sha1": "aaa"}}
    assert indexed_file_state.select_changed_chunks([chunk], {}) == [chunk]


def test_select_changed_chunks_drops_a_file_indexed_at_the_same_digest():
    chunk = {"metadata": {
        "repository": "kdk", "source_path": "map/x.js", "file_sha1": "aaa"}}
    indexed = {("kdk", "map/x.js"): "aaa"}
    assert indexed_file_state.select_changed_chunks([chunk], indexed) == []


def test_select_changed_chunks_keeps_a_file_whose_content_changed():
    chunk = {"metadata": {
        "repository": "kdk", "source_path": "map/x.js", "file_sha1": "bbb"}}
    indexed = {("kdk", "map/x.js"): "aaa"}
    assert indexed_file_state.select_changed_chunks([chunk], indexed) == [chunk]


def test_select_changed_chunks_distinguishes_repositories():
    # source_path is repo-relative, so the same path in another repo must not
    # be mistaken for an already indexed file.
    chunk = {"metadata": {
        "repository": "kano", "source_path": "map/x.js", "file_sha1": "aaa"}}
    indexed = {("kdk", "map/x.js"): "aaa"}
    assert indexed_file_state.select_changed_chunks([chunk], indexed) == [chunk]


def test_select_changed_chunks_keeps_every_chunk_of_a_changed_file():
    chunks = [
        {"metadata": {"repository": "kdk", "source_path": "map/x.js",
                      "file_sha1": "bbb", "chunk_index": 0}},
        {"metadata": {"repository": "kdk", "source_path": "map/x.js",
                      "file_sha1": "bbb", "chunk_index": 1}},
    ]
    indexed = {("kdk", "map/x.js"): "aaa"}
    assert indexed_file_state.select_changed_chunks(chunks, indexed) == chunks
