from ingestion.pipeline import change_detection as indexed_file_state


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

# Stub the two sources load_indexed_file_hashes reads: the chunk payloads
# and the file entries.
def _stub_index(monkeypatch, chunks=(), file_entries=()):
    monkeypatch.setattr(indexed_file_state.vectordb, "iter_chunk_payloads",
                        lambda: list(chunks))
    monkeypatch.setattr(indexed_file_state.vectordb, "iter_file_entry_payloads",
                        lambda: list(file_entries))


def test_load_indexed_file_hashes_keys_on_repository_and_path(
        monkeypatch):
    _stub_index(monkeypatch, chunks=[
        {"repo": "kdk", "path": "map/x.js", "file_sha1": "aaa"},
        {"repo": "kano", "path": "map/x.js", "file_sha1": "bbb"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {
        ("kdk", "map/x.js"): "aaa",
        ("kano", "map/x.js"): "bbb",
    }


def test_load_indexed_file_hashes_shares_one_entry_per_file(monkeypatch):
    # Every chunk of a file carries the same file_sha1, so many payloads
    # collapse into a single entry.
    _stub_index(monkeypatch, chunks=[
        {"repo": "kdk", "path": "map/x.js", "file_sha1": "aaa"},
        {"repo": "kdk", "path": "map/x.js", "file_sha1": "aaa"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {
        ("kdk", "map/x.js"): "aaa"}


def test_load_indexed_file_hashes_skips_payloads_without_a_digest(monkeypatch):
    _stub_index(monkeypatch, chunks=[
        {"repo": "kdk", "path": "map/x.js", "file_sha1": ""},
        {"repo": "kdk", "path": "map/y.js"},
    ])
    assert indexed_file_state.load_indexed_file_hashes() == {}


def test_load_indexed_file_hashes_is_empty_on_the_first_run(monkeypatch):
    _stub_index(monkeypatch)
    assert indexed_file_state.load_indexed_file_hashes() == {}


def test_load_indexed_file_hashes_ignores_the_file_entries(monkeypatch):
    # The chunks are the only source here. A file entry outlives the chunks
    # it described -- losing the code collection leaves every entry
    # standing -- so reading one would declare a file indexed that nothing
    # can retrieve any more, and no later run would ever rebuild it.
    _stub_index(
        monkeypatch,
        chunks=[],
        file_entries=[{"repo": "kdk", "path": "map/x.js",
                       "file_sha1": "aaa"}])
    assert indexed_file_state.load_indexed_file_hashes() == {}


# --- load_barren_file_hashes: the files that yield nothing to index -------

def test_load_barren_file_hashes_reads_the_file_entries(monkeypatch):
    # A .prettierrc.json produces nothing searchable, so it appears in no
    # chunk payload. Its entry is the only record that it was looked at;
    # without it the file comes back as new on every single run.
    _stub_index(
        monkeypatch,
        file_entries=[{"repo": "kdk", "path": ".prettierrc.json",
                       "file_sha1": "ccc"}])
    assert indexed_file_state.load_barren_file_hashes() == {
        ("kdk", ".prettierrc.json"): "ccc"}


def test_load_barren_file_hashes_skips_the_entries_without_a_digest(
        monkeypatch):
    # Only a file with nothing to index carries a digest on its entry, so
    # an entry without one belongs to a file the chunks already describe.
    _stub_index(
        monkeypatch,
        file_entries=[{"repo": "kdk", "path": "map/x.js", "file_sha1": ""},
                      {"repo": "kdk", "path": "map/y.js"}])
    assert indexed_file_state.load_barren_file_hashes() == {}


# --- get_file_key: the identity the whole index is keyed on ---------------

def test_get_file_key_splits_the_repository_from_the_path(tmp_path):
    path = tmp_path / "kdk" / "map" / "mixin.base-map.js"

    assert indexed_file_state.get_file_key(path, tmp_path) == (
        "kdk", "map/mixin.base-map.js")


def test_get_file_key_keeps_the_path_repository_relative(tmp_path):
    path = tmp_path / "kdk" / "packages" / "core" / "src" / "deep.js"

    assert indexed_file_state.get_file_key(path, tmp_path) == (
        "kdk", "packages/core/src/deep.js")


def test_get_file_key_distinguishes_the_same_path_in_two_repositories(tmp_path):
    # source_path is repo-relative, so the same path in another repo must not
    # be mistaken for an already indexed file.
    assert (indexed_file_state.get_file_key(tmp_path / "kdk/map/x.js", tmp_path)
            != indexed_file_state.get_file_key(tmp_path / "kano/map/x.js",
                                               tmp_path))


# --- hash_files: read once, hash once, never chunk to find out ------------

def test_hash_files_digests_every_scanned_file(tmp_path):
    first = _write(tmp_path, "kdk/map/x.js", "const a = 1")
    second = _write(tmp_path, "kdk/map/y.js", "const b = 2")

    assert indexed_file_state.hash_files([first, second]) == {
        first: indexed_file_state.compute_file_sha1("const a = 1"),
        second: indexed_file_state.compute_file_sha1("const b = 2"),
    }


def test_hash_files_digests_a_file_a_chunker_would_skip(tmp_path):
    # The digest comes from the raw text, so a file no chunker handles still
    # gets one; nothing here depends on the file being chunkable.
    path = _write(tmp_path, "kdk/scripts/deploy.py", "print('hi')")

    assert indexed_file_state.hash_files([path]) == {
        path: indexed_file_state.compute_file_sha1("print('hi')")}


def test_hash_files_returns_nothing_for_no_files():
    assert indexed_file_state.hash_files([]) == {}


# --- select_changed_files: the gate that keeps embedding incremental ------

def test_select_changed_files_keeps_a_file_that_is_not_indexed_yet(tmp_path):
    path = tmp_path / "kdk" / "map" / "x.js"

    assert indexed_file_state.select_changed_files(
        {path: "aaa"}, tmp_path, {}) == [path]


def test_select_changed_files_drops_a_file_indexed_at_the_same_digest(
        tmp_path):
    path = tmp_path / "kdk" / "map" / "x.js"
    indexed = {("kdk", "map/x.js"): "aaa"}

    assert indexed_file_state.select_changed_files(
        {path: "aaa"}, tmp_path, indexed) == []


def test_select_changed_files_keeps_a_file_whose_content_changed(tmp_path):
    path = tmp_path / "kdk" / "map" / "x.js"
    indexed = {("kdk", "map/x.js"): "aaa"}

    assert indexed_file_state.select_changed_files(
        {path: "bbb"}, tmp_path, indexed) == [path]


def test_select_changed_files_distinguishes_repositories(tmp_path):
    path = tmp_path / "kano" / "map" / "x.js"
    indexed = {("kdk", "map/x.js"): "aaa"}

    assert indexed_file_state.select_changed_files(
        {path: "aaa"}, tmp_path, indexed) == [path]


def test_select_changed_files_keeps_everything_on_the_first_run(tmp_path):
    # Nothing indexed yet, so every lookup misses and the whole corpus is
    # selected -- a full rebuild needs no branch of its own.
    file_hashes = {tmp_path / "kdk" / "map" / f"x{i}.js": "aaa"
                   for i in range(3)}

    assert (indexed_file_state.select_changed_files(file_hashes, tmp_path, {})
            == list(file_hashes))


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Write a workspace-relative file, creating parent directories as needed.
def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
