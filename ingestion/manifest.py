"""Incremental indexing: skip files already indexed at the same content.

Re-running the pipeline re-embeds every file it is given, which is wasteful
when most files are unchanged -- embedding is the expensive step. This
module reconstructs, from the Qdrant payloads, which (repository,
source_path) is already indexed and at what whole-file hash, so the
pipeline can drop chunks whose file is unchanged before embedding them.

The manifest is not stored separately: every point already carries
repository, source_path and file_sha1 in its payload (see points.py), so
scrolling the collection rebuilds it.
"""

import hashlib

from utils.vectordb import client as vectordb


# Hex SHA-1 of a whole file's text. Stamped on every chunk of the file so
# the manifest can tell, per file, whether its content changed.
def file_sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Rebuild {(repository, source_path): file_sha1} from the indexed points.
# Empty on the first run, when nothing is indexed yet.
def load():
    indexed = {}
    for payload in vectordb.iter_payloads():
        digest = payload.get("file_sha1", "")
        if not digest:
            continue
        key = (payload.get("repository", ""), payload.get("source_path", ""))
        indexed[key] = digest
    return indexed


# Keep only chunks whose file is new or changed versus the indexed manifest.
def select_changed(chunks, indexed):
    return [chunk for chunk in chunks if _changed(chunk["metadata"], indexed)]


# A file is unchanged only if its (repository, source_path) is indexed at
# the exact same file hash; anything else (new, edited) counts as changed.
def _changed(metadata, indexed):
    key = (metadata.get("repository", ""), metadata["source_path"])
    return indexed.get(key) != metadata.get("file_sha1", "")
