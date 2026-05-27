"""Internal engine for walking the file tree and applying filter rules.

This module contains the heavy lifting of the scanning process. It uses 
os.walk for efficiency and applies multiple layers of checks (path, 
extension, pattern, size, and content-aware minification detection).
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import FileRecord, FilterConfig, ScanResult


def scan_tree(root: Path, config: FilterConfig, profile_name: str) -> ScanResult:
    """Recursively crawls a directory and builds a ScanResult.
    
    Optimizes performance by pruning excluded directories during the walk 
    and avoiding unnecessary file reads until the final check stage.
    """
    result = ScanResult(root_dir=root, config=config, profile_name=profile_name)
    
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # followlinks=True so the notebook-style data/ symlinks to kalisio
        # sibling repos (kdk, kano, etc.) are traversed. Production would
        # never set this; this is an experiment-side accommodation.
        # In-place modification of dirnames prunes them from os.walk
        dirnames[:] = [d for d in dirnames if d not in config.excluded_dirs]
        
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                # Get file size first as it's a cheap metadata operation
                size = fpath.stat().st_size
            except OSError:
                continue

            rel = str(fpath.relative_to(root))
            ext = fpath.suffix.lower()
            parts = Path(rel).parts
            zone = parts[0] if len(parts) > 1 else "(root)"

            rec = FileRecord(path=fpath, rel_path=rel, extension=ext, size=size, zone=zone)
            
            # Layered filter check
            reason = check_exclusion(rec, config)
            if reason:
                rec.exclude_reason = reason
                result.excluded.append(rec)
            else:
                result.included.append(rec)

    # Sort for deterministic output across runs
    result.included.sort(key=lambda r: r.rel_path)
    result.excluded.sort(key=lambda r: r.rel_path)
    return result


def check_exclusion(record: FileRecord, config: FilterConfig) -> str:
    """Evaluates a file against all active filter rules.
    
    Checks are ordered from cheapest (filename/extension) to most 
    expensive (reading file content for minification detection).
    """
    # 1. Exact filename match (e.g., 'yarn.lock')
    if record.path.name in config.excluded_filenames:
        return "excluded_filename"

    # 2. Extension match (e.g., '.png')
    if record.extension in config.excluded_extensions:
        return "binary_or_non_text"

    # 3. Empty extension (usually system files or directories)
    if not record.extension:
        return "no_extension"

    # 4. Regex patterns (e.g., '.min.js')
    for pattern in config.excluded_patterns:
        if pattern.search(record.path.name):
            return "pattern_match"

    # 5. File size limit
    if config.max_file_size and record.size > config.max_file_size:
        return "file_too_large"

    # 6. Content-aware check: line length (detects minified code or JSON data)
    if config.max_line_length and record.size > config.max_line_length:
        try:
            with open(record.path, errors="ignore") as fh:
                for line in fh:
                    if len(line) > config.max_line_length:
                        return "long_line_minified_or_data"
        except OSError:
            pass

    # 7. Whitelist check (if active)
    if config.included_extensions and record.extension not in config.included_extensions:
        return "extension_not_in_whitelist"

    return ""  # Included
