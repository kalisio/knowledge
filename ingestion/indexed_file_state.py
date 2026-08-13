import hashlib

import utils.vectordb as vectordb


# Hex SHA-1 of a whole file's text. Stamped on every chunk of the file so
# the indexed state can tell, per file, whether its content changed.
def compute_file_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# A file's identity everywhere in the index: the first path segment under
# `workspace` is its repository, the rest its repo-relative source path.
def file_key(path, workspace):
    parts = path.relative_to(workspace).parts
    return parts[0], "/".join(parts[1:])


# {Path: file_sha1} for the scanned files. Reading and hashing is all it
# takes to know which files changed, so the corpus is never chunked just to
# answer that question -- only the files this returns as changed are.
def hash_files(files):
    return {path: compute_file_sha1(
                path.read_text(encoding="utf-8", errors="ignore"))
            for path in files}


# Rebuild {(repository, source_path): file_sha1} from the indexed entries.
# Empty on the first run, when nothing is indexed yet.
def load_indexed_file_hashes():
    indexed = {}
    for payload in vectordb.iter_payloads():
        digest = payload.get("file_sha1", "")
        if not digest:
            continue
        key = (payload.get("repository", ""), payload.get("source_path", ""))
        indexed[key] = digest
    return indexed


# The scanned files that are new or whose content changed. A file is
# unchanged only if its (repository, source_path) is indexed at the exact
# same digest; anything else (new, edited) counts as changed.
def select_changed_files(file_hashes, workspace, indexed):
    return [path for path, digest in file_hashes.items()
            if indexed.get(file_key(path, workspace)) != digest]


# (repository, source_path) pairs indexed but no longer scanned.
def find_deleted_files(indexed, current_files):
    return set(indexed) - set(current_files)
