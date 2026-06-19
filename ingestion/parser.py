"""Scan the local Kalisio repositories for files to ingest.

Walk a repository, skip noise (VCS, dependencies, build output, caches),
and keep only the file types we have chunkers for (see ingestion/chunks/),
dropping minified bundles, lock/metadata files, and oversized files. Rules
are adapted from the file-filtering experiment (corpus_filter/), pared down
to what production needs.
"""

import os
import re
from pathlib import Path


# File types we index — one per chunker in ingestion/chunks/.
INDEXED_SUFFIXES = {".md", ".js", ".mjs", ".cjs", ".vue", ".json"}

# Directory names to skip entirely (VCS, deps, build output, caches, IDE).
IGNORED_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components", ".yarn", ".pnpm-store",
    "dist", "build", ".output", ".next", ".nuxt", ".vite",
    "coverage", ".nyc_output", ".c8",
    "__pycache__", ".cache", ".parcel-cache", ".turbo",
    ".github", ".gitlab", ".vscode", ".idea",
}

# Filenames to skip even when their suffix is indexed (metadata/noise).
IGNORED_FILENAMES = {
    "package.json", "package-lock.json",
    "CHANGELOG.md", "changelog.md", "CHANGES.md",
    "LICENSE.md",
}

# Generated/minified/bundled files (e.g. foo.min.js, vendor.bundle.js).
IGNORED_PATTERN = re.compile(r"\.(min|bundle|chunk)\.\w+$|-lock\.json$")

# Skip files larger than this; big files are usually data, not source.
MAX_FILE_SIZE = 100_000


# Walk `root` (skipping IGNORED_DIRS) and return the files we want to index.
def scan(root):
    kept = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk does not descend into ignored dirs.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if _is_indexed(path):
                kept.append(path)
    return kept


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# True when `path` passes the suffix, filename, pattern, and size checks.
def _is_indexed(path):
    if path.suffix.lower() not in INDEXED_SUFFIXES:
        return False
    if path.name in IGNORED_FILENAMES:
        return False
    if IGNORED_PATTERN.search(path.name):
        return False
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return True
