"""Scan the local Kalisio repositories for files to ingest.
"""

import os
import sys
from collections import Counter
from pathlib import Path


# Directories never worth indexing (VCS metadata, installed dependencies).
IGNORED_DIRS = {".git", "node_modules"}

# File types we index — one per chunker in ingestion/chunks/.
INDEXED_SUFFIXES = {".md", ".js", ".mjs", ".cjs", ".vue", ".json"}


# Walk `root`, skip IGNORED_DIRS, and return the files we want to index.
def scan(root):
    kept = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk does not descend into ignored dirs.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in INDEXED_SUFFIXES:
                kept.append(path)
    return kept


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = Path(__file__).resolve().parents[1]
    files = scan(root)
    by_suffix = Counter(path.suffix.lower() for path in files)
    print(f"scanned root : {root}")
    print(f"kept files   : {len(files)}")
    print(f"by suffix    : {dict(sorted(by_suffix.items()))}")
    for path in files[:5]:
        print(f"  {path}")
