import hashlib
from functools import lru_cache
from pathlib import Path

import ingestion.services.vectordb as vectordb


# Hex SHA-1 of a whole file's text. Stamped on every chunk of the file so
# the indexed state can tell, per file, whether its content changed.
def compute_file_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# A file's identity everywhere in the index: the repository holding it, and
# its path inside that repository. The repository is found by walking up to
# the nearest .git, because a workspace nests them one level down per
# organisation (kalisio/kdk, irsn/planet) while a hand-made one does not.
def get_file_key(path, workspace):
    repository = _repository_of(path.parent, Path(workspace))
    if repository is None:
        # Nothing above the file is a repository: fall back on the first
        # segment, which is what a flat workspace means.
        parts = path.relative_to(workspace).parts
        return parts[0], "/".join(parts[1:])
    return repository.name, path.relative_to(repository).as_posix()


# {Path: file_sha1} for the scanned files. Reading and hashing is all it
# takes to know which files changed, so the corpus is never chunked just to
# answer that question -- only the files this returns as changed are.
def hash_files(files):
    return {path: compute_file_sha1(
                path.read_text(encoding="utf-8", errors="ignore"))
            for path in files}


# Rebuild {(repository, path): file_sha1} from the indexed entries.
# Empty on the first run, when nothing is indexed yet.
def load_indexed_file_hashes():
    indexed = {}
    for payload in vectordb.iter_payloads():
        digest = payload.get("file_sha1", "")
        if not digest:
            continue
        key = (payload.get("repo", ""), payload.get("path", ""))
        indexed[key] = digest
    return indexed


# The scanned files that are new or whose content changed. A file is
# unchanged only if its (repository, path) is indexed at the exact
# same digest; anything else (new, edited) counts as changed.
def select_changed_files(file_hashes, workspace, indexed):
    return [path for path, digest in file_hashes.items()
            if indexed.get(get_file_key(path, workspace)) != digest]


# (repository, path) pairs indexed but no longer scanned.
def find_deleted_files(indexed, current_files):
    return set(indexed) - set(current_files)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The nearest directory at or above `directory` holding a .git, stopping at
# the workspace. Cached: this runs once per file of the corpus.
@lru_cache(maxsize=8192)
def _repository_of(directory, workspace):
    if directory == workspace or workspace not in directory.parents:
        return None
    if (directory / ".git").exists():
        return directory
    return _repository_of(directory.parent, workspace)
