import subprocess

from ingestion.file_scanner import scan_indexable_files


# --- supported extensions -------------------------------------------------

def test_scan_keeps_the_supported_extensions(tmp_path, ingestion_env):
    ingestion_env()
    for name in ("a.md", "b.js", "c.mjs", "d.cjs", "e.vue", "f.json"):
        _write(tmp_path, f"kdk/{name}")

    assert _kept_names(tmp_path) == {
        "a.md", "b.js", "c.mjs", "d.cjs", "e.vue", "f.json"}


def test_scan_drops_unsupported_extensions(tmp_path, ingestion_env):
    ingestion_env()
    _write(tmp_path, "kdk/script.py")
    _write(tmp_path, "kdk/notes.txt")
    _write(tmp_path, "kdk/logo.png")
    _write(tmp_path, "kdk/keep.md")

    assert _kept_names(tmp_path) == {"keep.md"}


def test_scan_matches_extensions_case_insensitively(tmp_path, ingestion_env):
    ingestion_env()
    _write(tmp_path, "kdk/README.MD")
    _write(tmp_path, "kdk/Component.VUE")

    assert _kept_names(tmp_path) == {"README.MD", "Component.VUE"}


def test_scan_honours_a_narrowed_extension_list(tmp_path, ingestion_env):
    ingestion_env(SUPPORTED_FILE_EXTENSIONS="md")
    _write(tmp_path, "kdk/keep.md")
    _write(tmp_path, "kdk/drop.js")

    assert _kept_names(tmp_path) == {"keep.md"}


# --- ignored directories --------------------------------------------------

def test_scan_does_not_descend_into_ignored_directories(
        tmp_path, ingestion_env):
    # Vendored code dwarfs the real corpus, so this filter is what keeps the
    # index about Kalisio rather than about its dependencies.
    ingestion_env()
    _write(tmp_path, "kdk/node_modules/lib/index.js")
    _write(tmp_path, "kdk/dist/bundle.js")
    _write(tmp_path, "kdk/.git/COMMIT_EDITMSG.md")
    _write(tmp_path, "kdk/src/real.js")

    assert _kept_names(tmp_path) == {"real.js"}


def test_scan_ignores_directories_at_any_depth(tmp_path, ingestion_env):
    ingestion_env()
    _write(tmp_path, "kdk/packages/core/node_modules/dep/a.js")
    _write(tmp_path, "kdk/packages/core/src/b.js")

    assert _kept_names(tmp_path) == {"b.js"}


# --- ignored file names and generated artefacts ---------------------------

def test_scan_drops_ignored_file_names(tmp_path, ingestion_env):
    ingestion_env()
    _write(tmp_path, "kdk/package.json")
    _write(tmp_path, "kdk/CHANGELOG.md")
    _write(tmp_path, "kdk/LICENSE.md")
    _write(tmp_path, "kdk/README.md")

    assert _kept_names(tmp_path) == {"README.md"}


def test_scan_drops_generated_artefacts(tmp_path, ingestion_env):
    # Minified and lock files are machine output: they embed badly and would
    # crowd out hand-written code in the results.
    ingestion_env()
    _write(tmp_path, "kdk/app.min.js")
    _write(tmp_path, "kdk/vendor.bundle.js")
    _write(tmp_path, "kdk/main.chunk.js")
    _write(tmp_path, "kdk/yarn-lock.json")
    _write(tmp_path, "kdk/app.js")

    assert _kept_names(tmp_path) == {"app.js"}


# --- size limit -----------------------------------------------------------

def test_scan_drops_files_over_the_size_limit(tmp_path, ingestion_env):
    ingestion_env(MAX_FILE_SIZE=50)
    _write(tmp_path, "kdk/small.md", "x" * 10)
    _write(tmp_path, "kdk/huge.md", "x" * 100)

    assert _kept_names(tmp_path) == {"small.md"}


def test_scan_keeps_a_file_exactly_at_the_size_limit(tmp_path, ingestion_env):
    ingestion_env(MAX_FILE_SIZE=50)
    _write(tmp_path, "kdk/edge.md", "x" * 50)

    assert _kept_names(tmp_path) == {"edge.md"}


# --- only what git tracks is corpus ---------------------------------------

def test_scan_skips_untracked_files(tmp_path, ingestion_env):
    # Untracked files are local noise (caches, scratch output, secrets):
    # what a repository does not track is not corpus.
    ingestion_env()
    _write(tmp_path, "kdk/tracked.md")
    _track_repositories(tmp_path)
    _write(tmp_path, "kdk/untracked.md")

    assert ({path.name for path in scan_indexable_files(tmp_path)}
            == {"tracked.md"})


def test_scan_skips_a_directory_that_is_not_a_repository(
        tmp_path, ingestion_env):
    # A workspace can hold non-repository directories (key stores, database
    # data); nothing in them may reach the index.
    ingestion_env()
    _write(tmp_path, "age/keys.md")

    assert scan_indexable_files(tmp_path) == []


def test_scan_skips_a_tracked_file_missing_from_disk(tmp_path, ingestion_env):
    ingestion_env()
    path = _write(tmp_path, "kdk/gone.md")
    _track_repositories(tmp_path)
    path.unlink()

    assert scan_indexable_files(tmp_path) == []


# --- empty workspace ------------------------------------------------------

def test_scan_returns_nothing_for_an_empty_workspace(tmp_path, ingestion_env):
    ingestion_env()
    assert scan_indexable_files(tmp_path) == []


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Write a workspace-relative file, creating parent directories as needed.
def _write(root, relative, text="x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# Turn each top-level directory into a git repository tracking its current
# files, as k-clone leaves the workspace.
def _track_repositories(root):
    for child in sorted(root.iterdir()):
        if child.is_dir():
            _git(child, "init", "-q")
            _git(child, "add", "-A")


# Run a git command in a repository, quietly.
def _git(repo_dir, *args):
    subprocess.run(["git", "-C", str(repo_dir), *args],
                   capture_output=True, check=True)


# The names of the files the scanner keeps, as a set for order-free asserts.
def _kept_names(root):
    _track_repositories(root)
    return {path.name for path in scan_indexable_files(root)}
