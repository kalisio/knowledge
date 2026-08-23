import subprocess

from ingestion.services.scanner import (find_repositories,
                                        scan_indexable_files)


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


# --- the workspace layout --------------------------------------------------
# DEVELOPMENT_DIR holds one directory per organisation, and each of those
# holds the cloned repositories, so a repository sits two levels down.

def test_a_repository_under_an_organisation_is_scanned(tmp_path, ingestion_env):
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path))
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")

    scanned = scan_indexable_files(tmp_path)

    assert [path.name for path in scanned] == ["base.js"]


def test_every_organisation_is_scanned(tmp_path, ingestion_env):
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path))
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")
    _repository(tmp_path / "irsn" / "planet", "docs/guide.md")
    _repository(tmp_path / "airbus" / "Gift-adsb", "index.js")

    scanned = scan_indexable_files(tmp_path)

    assert sorted(path.name for path in scanned) == [
        "base.js", "guide.md", "index.js"]


def test_a_repository_sitting_directly_under_the_root_is_scanned(
        tmp_path, ingestion_env):
    # A workspace laid out by hand keeps the repositories one level up.
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path))
    _repository(tmp_path / "kdk", "map/base.js")

    assert [path.name for path in scan_indexable_files(tmp_path)] == ["base.js"]


def test_the_repositories_are_reported_with_their_path(tmp_path):
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")
    _repository(tmp_path / "irsn" / "planet", "index.js")

    found = find_repositories(tmp_path)

    assert {repo.name for repo in found} == {"kdk", "planet"}
    assert all(repo.is_dir() for repo in found)


def test_a_directory_that_is_not_a_repository_is_not_reported(tmp_path):
    (tmp_path / "kalisio" / "notes").mkdir(parents=True)
    (tmp_path / "kalisio" / "notes" / "a.md").write_text("# a")

    assert find_repositories(tmp_path) == []


def test_the_age_key_directory_holds_no_repository(tmp_path):
    # DEVELOPMENT_DIR also holds age/ (encryption keys), mongo/ and redis/.
    (tmp_path / "age").mkdir()
    (tmp_path / "age" / "keys.txt").write_text("secret")

    assert find_repositories(tmp_path) == []
    assert scan_indexable_files(tmp_path) == []


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# A git repository at `path` tracking one file.
def _repository(path, source_path):
    import subprocess
    file_path = path / source_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("const a = 1\n")
    subprocess.run(["git", "-C", str(path), "init", "-q"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(path), "add", "-A"],
                   capture_output=True, check=True)
    return path


# --- narrowing the scan to some repositories -------------------------------

def test_every_repository_is_scanned_by_default(tmp_path, ingestion_env):
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path))
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")
    _repository(tmp_path / "kalisio" / "kano", "index.js")

    assert {repo.name for repo in find_repositories(tmp_path)} == {"kdk", "kano"}


def test_the_scan_can_be_narrowed_to_one_repository(tmp_path, ingestion_env):
    # What a developer runs: index kdk alone instead of the whole ecosystem.
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path), INDEXED_REPOSITORIES="kdk")
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")
    _repository(tmp_path / "kalisio" / "kano", "index.js")

    assert [repo.name for repo in find_repositories(tmp_path)] == ["kdk"]
    assert [path.name for path in scan_indexable_files(tmp_path)] == ["base.js"]


def test_several_repositories_can_be_named(tmp_path, ingestion_env):
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path),
                  INDEXED_REPOSITORIES="kdk, kano")
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")
    _repository(tmp_path / "kalisio" / "kano", "index.js")
    _repository(tmp_path / "irsn" / "planet", "app.js")

    assert {repo.name for repo in find_repositories(tmp_path)} == {"kdk", "kano"}


def test_a_named_repository_that_is_not_cloned_is_skipped(
        tmp_path, ingestion_env):
    # "only kdk, if it is there": a developer who has not cloned it gets an
    # empty run, not a failure.
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path), INDEXED_REPOSITORIES="kdk")
    _repository(tmp_path / "kalisio" / "kano", "index.js")

    assert find_repositories(tmp_path) == []
    assert scan_indexable_files(tmp_path) == []


def test_an_empty_setting_means_every_repository(tmp_path, ingestion_env):
    ingestion_env(DEVELOPMENT_DIR=str(tmp_path), INDEXED_REPOSITORIES="")
    _repository(tmp_path / "kalisio" / "kdk", "map/base.js")

    assert [repo.name for repo in find_repositories(tmp_path)] == ["kdk"]
