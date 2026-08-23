from ingestion import chunkers
from ingestion.services.state import compute_file_sha1

JAVASCRIPT = "export function center () {\n  return [0, 0]\n}\n"
MARKDOWN = "# Title\n\nSome prose about the map.\n"


# --- dispatch by extension ------------------------------------------------

def test_chunk_files_dispatches_each_extension_to_a_chunker(tmp_path):
    _write(tmp_path, "kdk/map/base.js", JAVASCRIPT)
    _write(tmp_path, "kdk/docs/guide.md", MARKDOWN)

    chunks = chunkers.chunk_files(_files(tmp_path), tmp_path)

    assert {chunk["metadata"]["path"] for chunk in chunks} == {
        "map/base.js", "docs/guide.md"}


def test_chunk_files_skips_extensions_without_a_chunker(tmp_path):
    # The incremental path feeds this function straight from git, which knows
    # nothing about the scanner's filters, so unsupported files arrive here.
    _write(tmp_path, "kdk/scripts/deploy.py", "print('hi')")
    _write(tmp_path, "kdk/map/base.js", JAVASCRIPT)

    chunks = chunkers.chunk_files(_files(tmp_path), tmp_path)

    assert all(chunk["metadata"]["path"] == "map/base.js"
               for chunk in chunks)


def test_chunk_files_dispatches_on_a_lowercased_extension(tmp_path):
    _write(tmp_path, "kdk/docs/GUIDE.MD", MARKDOWN)

    chunks = chunkers.chunk_files(_files(tmp_path), tmp_path)

    assert chunks


def test_chunk_files_returns_nothing_for_no_files(tmp_path):
    assert chunkers.chunk_files([], tmp_path) == []


# --- repo and path: the key everything else uses -------------

def test_chunk_files_splits_the_repository_from_the_path(tmp_path):
    # delete_file() and the indexed state both key on this pair, so a wrong
    # split silently orphans a file's chunks instead of replacing them.
    _write(tmp_path, "kdk/map/mixin.base-map.js", JAVASCRIPT)

    chunk = chunkers.chunk_files(_files(tmp_path), tmp_path)[0]

    assert chunk["metadata"]["repo"] == "kdk"
    assert chunk["metadata"]["path"] == "map/mixin.base-map.js"


def test_chunk_files_keeps_the_path_repository_relative(tmp_path):
    _write(tmp_path, "kdk/packages/core/src/deep.js", JAVASCRIPT)

    chunk = chunkers.chunk_files(_files(tmp_path), tmp_path)[0]

    assert chunk["metadata"]["repo"] == "kdk"
    assert chunk["metadata"]["path"] == "packages/core/src/deep.js"


def test_chunk_files_distinguishes_the_same_path_in_two_repositories(tmp_path):
    _write(tmp_path, "kdk/map/base.js", JAVASCRIPT)
    _write(tmp_path, "kano/map/base.js", JAVASCRIPT)

    chunks = chunkers.chunk_files(_files(tmp_path), tmp_path)

    assert {chunk["metadata"]["repo"] for chunk in chunks} == {
        "kdk", "kano"}


# --- file_sha1: the gate on the next run ----------------------------------

def test_chunk_files_stamps_the_digest_of_the_whole_file(tmp_path):
    _write(tmp_path, "kdk/map/base.js", JAVASCRIPT)

    chunk = chunkers.chunk_files(_files(tmp_path), tmp_path)[0]

    assert chunk["metadata"]["file_sha1"] == compute_file_sha1(JAVASCRIPT)


def test_chunk_files_stamps_every_chunk_of_a_file_with_the_same_digest(
        tmp_path):
    # The indexed state compares per file, not per chunk: one chunk carrying
    # a different digest would re-embed a file that never changed.
    _write(tmp_path, "kdk/map/base.js", JAVASCRIPT * 40)

    chunks = chunkers.chunk_files(_files(tmp_path), tmp_path)

    assert len(chunks) > 1
    assert len({chunk["metadata"]["file_sha1"] for chunk in chunks}) == 1


def test_chunk_files_changes_the_digest_when_the_content_changes(tmp_path):
    path = _write(tmp_path, "kdk/map/base.js", JAVASCRIPT)
    before = chunkers.chunk_files([path], tmp_path)[0]

    path.write_text(JAVASCRIPT.replace("[0, 0]", "[1, 1]"))
    after = chunkers.chunk_files([path], tmp_path)[0]

    assert before["metadata"]["file_sha1"] != after["metadata"]["file_sha1"]


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Write a workspace-relative file, creating parent directories as needed.
def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# Every file under the workspace, as the callers of chunk_files pass them.
def _files(root):
    return sorted(path for path in root.rglob("*") if path.is_file())
