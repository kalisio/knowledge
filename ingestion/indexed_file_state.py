import hashlib

import utils.vectordb as vectordb


# Hex SHA-1 of a whole file's text. Stamped on every chunk of the file so
# the indexed state can tell, per file, whether its content changed.
def compute_file_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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


# Keep only chunks whose file is new or changed versus the indexed state.
def select_changed_chunks(chunks, indexed):
    return [chunk for chunk in chunks if _changed(chunk["metadata"], indexed)]


# (repository, source_path) pairs indexed but no longer scanned.
def find_deleted_files(indexed, current_files):
    return set(indexed) - set(current_files)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# A file is unchanged only if its (repository, source_path) is indexed at
# the exact same file hash; anything else (new, edited) counts as changed.
def _changed(metadata, indexed):
    key = (metadata.get("repository", ""), metadata["source_path"])
    return indexed.get(key) != metadata.get("file_sha1", "")
