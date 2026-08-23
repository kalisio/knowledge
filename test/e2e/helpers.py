"""Building blocks for the end-to-end suite.

Everything here is deterministic and offline: the workspace is a real set of
git repositories built on a tmp_path, the embedding model is replaced by a
lexical stand-in, and Qdrant is the only external service involved.
"""

import hashlib
import math
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

import ingestion.services.vectordb as vectordb
import api.services.vectordb as api_vectordb
from api.services.embeddings import QUERY_PREFIX

# Throwaway collections; the "zz_" prefix keeps them apart from the
# collections a developer ingests into locally.
CODE_COLLECTION = "zz_e2e_code"
METADATA_COLLECTION = "zz_e2e_metadata"
FILES_COLLECTION = f"{CODE_COLLECTION}_files"

# Dimension of the lexical stand-in vectors (see lexical_vector). Wide
# enough that two unrelated files do not collide into the same buckets.
VECTOR_SIZE = 1024

# The four supported file types, each mapped to its sample in
# test/data/sources and to the place it takes in the workspace: two
# repositories, so `repo` is exercised.
SAMPLES = {
    "kdk/docs/catalog/guide.md": "guide.md",
    "kdk/core/client/geolocation.js": "geolocation.js",
    "kano/client/components/KLayerList.vue": "KLayerList.vue",
    "kano/client/i18n/en.json": "en.json",
}

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"

_TOKEN = re.compile(r"[a-z0-9]+")


# An ISO date `days` ago, for placing a commit relative to the history window.
def days_ago(days):
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                         time.gmtime(time.time() - days * 86400))


# ---------------------------------------------------------------------------
# QDRANT AVAILABILITY
# ---------------------------------------------------------------------------


# True when Qdrant answers at QDRANT_URL. The whole suite drives the real
# vector database, so every test is skipped when it is unreachable.
def qdrant_reachable():
    url = os.environ.get("QDRANT_URL")
    if not url:
        return False
    try:
        urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2)
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_reachable(), reason="needs a running Qdrant (QDRANT_URL)")


# ---------------------------------------------------------------------------
# EMBEDDING STAND-IN
# ---------------------------------------------------------------------------


# A deterministic bag-of-words vector: cosine similarity between two of them
# is real lexical similarity, so a query actually ranks the chunk that talks
# about it first. That is what makes retrieval assertions meaningful without
# loading the real (hundreds of MB) sentence-transformer model.
def lexical_vector(text):
    counts = [0.0] * VECTOR_SIZE
    for token in _TOKEN.findall(text.lower()):
        counts[_token_bucket(token)] += 1.0
    norm = math.sqrt(sum(count * count for count in counts))
    if norm == 0.0:
        # An empty/symbol-only text still needs a unit vector: Qdrant rejects
        # a zero vector under cosine distance.
        return [1.0] + [0.0] * (VECTOR_SIZE - 1)
    return [count / norm for count in counts]


# Document encoder stand-in for utils.embeddings.encode_batch.
def fake_encode_batch(texts):
    return [lexical_vector(text) for text in texts]


# Query encoder stand-in for utils.embeddings.encode. The real encoder is
# asymmetric (it prepends QUERY_PREFIX); the prefix is stripped back off here
# so the stand-in stays comparable with the document vectors, while a test
# can still assert the prefix reached the encoder.
def fake_encode(text):
    return lexical_vector(text.replace(QUERY_PREFIX, ""))


# ---------------------------------------------------------------------------
# WORKSPACE
# ---------------------------------------------------------------------------


# A workspace laid out the way k-clone leaves one: git repositories sitting
# directly under DEVELOPMENT_DIR. Only tracked files are scanned, so every
# helper here goes through git.
class Workspace:
    def __init__(self, root):
        self.root = Path(root)

    # Absolute path of a workspace-relative file.
    def path(self, source_path):
        return self.root / source_path


    # Write a workspace-relative file and commit it, optionally at a chosen
    # date so a test can place a commit inside or outside the history window.
    def commit(self, source_path, text, message="chore: change", date=None):
        path = self.path(source_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        repo = self._ensure_repo(source_path)
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message, date=date)
        return path

    # Write raw bytes and commit them (invalid UTF-8, BOM, ...).
    def commit_bytes(self, source_path, data, message="chore: change"):
        path = self.path(source_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        repo = self._ensure_repo(source_path)
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", message)
        return path

    # Delete a workspace-relative file and commit the deletion.
    def remove(self, source_path, message="chore: remove"):
        repo = self._repo_dir(source_path)
        relative = str(Path(source_path).relative_to(repo.name))
        self.git(repo, "rm", "-q", relative)
        self.git(repo, "commit", "-q", "-m", message)

    # Install the four samples, one per supported file type.
    def install_samples(self, message="feat: import the corpus", date=None):
        for source_path, sample in SAMPLES.items():
            self.commit(source_path, read_sample(sample), message, date=date)

    # Run a git command in `repo`, failing loudly, optionally at a fixed
    # date so commits can be placed in time.
    def git(self, repo, *args, date=None):
        env = dict(os.environ)
        if date:
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, check=True, env=env)

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    # Repository directory of a workspace-relative path.
    def _repo_dir(self, source_path):
        return self.root / Path(source_path).parts[0]

    # Create the repository on first use, with a deterministic identity.
    def _ensure_repo(self, source_path):
        repo = self._repo_dir(source_path)
        repo.mkdir(parents=True, exist_ok=True)
        if not (repo / ".git").exists():
            self.git(repo, "init", "-q", "-b", "main")
            self.git(repo, "config", "user.email", "test@kalisio.com")
            self.git(repo, "config", "user.name", "test")
        return repo


# Read one of the sample files shipped in test/samples.
def read_sample(name):
    return (_SAMPLES_DIR / name).read_text(encoding="utf-8")


# (repository, repo-relative path) key of a workspace-relative path.
def file_key(source_path):
    repository, relative = source_path.split("/", 1)
    return repository, relative


# SHA-1 of a text, the digest the pipeline stamps on every chunk of a file.
def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# INDEX INSPECTION
# ---------------------------------------------------------------------------


# Every point stored in the throwaway code collection.
def code_points():
    client = vectordb._get_qdrant_client()
    records, next_page = client.scroll(
        collection_name=CODE_COLLECTION, limit=1000, with_payload=True,
        with_vectors=False)
    # A next page would mean the assertions only see part of the collection.
    assert next_page is None, "the test corpus outgrew the one-shot scroll"
    return records


# Stored points grouped by (repository, source_path).
def points_by_file():
    grouped = {}
    for record in code_points():
        key = (record.payload["repo"], record.payload["path"])
        grouped.setdefault(key, []).append(record)
    return grouped


# The commit history stored for one workspace-relative file, read the way
# the API reads it.
def history_of(source_path):
    repository, path = file_key(source_path)
    return api_vectordb.get_commit_histories([(repository, path)]).get(
        (repository, path), [])


# The chunks of one workspace-relative file, ordered by chunk_index.
def chunks_of(source_path):
    key = file_key(source_path)
    points = points_by_file().get(key, [])
    return sorted((point.payload for point in points),
                  key=lambda payload: payload["chunk_index"])


# Drop every throwaway collection if it exists.
def drop_collections():
    for name in (CODE_COLLECTION, METADATA_COLLECTION, FILES_COLLECTION):
        if vectordb.check_collection_exists(name):
            vectordb.remove_collection(name)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Stable bucket for a token: hash() is salted per process, sha1 is not.
def _token_bucket(token):
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % VECTOR_SIZE
