import os
import re
from pathlib import Path

from ingestion.config import get_ingestion_config


# Get the files to index under `root`.
def scan_indexable_files(root):
    config = get_ingestion_config()
    supported_file_extensions = config.supported_file_extensions.split(",")
    ignored_directories = config.ignored_directories.split(",")
    ignored_filenames = config.ignored_filenames.split(",")
    ignored_file_pattern = re.compile(config.ignored_file_pattern)
    kept = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored_directories]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lstrip(".").lower() not in supported_file_extensions:
                continue
            if filename in ignored_filenames:
                continue
            if ignored_file_pattern.search(filename):
                continue
            if path.stat().st_size > config.max_file_size:
                continue
            kept.append(path)
    return kept
