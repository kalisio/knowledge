from __future__ import annotations

import os
from pathlib import Path

from .models import FileRecord, FilterConfig, ScanResult


def scan_tree(root: Path, config: FilterConfig, profile_name: str) -> ScanResult:
    result = ScanResult(root_dir=root, config=config, profile_name=profile_name)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.excluded_dirs]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            rel = str(fpath.relative_to(root))
            ext = fpath.suffix.lower()
            parts = Path(rel).parts
            zone = parts[0] if len(parts) > 1 else "(root)"

            rec = FileRecord(path=fpath, rel_path=rel, extension=ext, size=size, zone=zone)
            reason = check_exclusion(rec, config)
            if reason:
                rec.exclude_reason = reason
                result.excluded.append(rec)
            else:
                result.included.append(rec)

    result.included.sort(key=lambda r: r.rel_path)
    result.excluded.sort(key=lambda r: r.rel_path)
    return result


def check_exclusion(record: FileRecord, config: FilterConfig) -> str:
    if record.path.name in config.excluded_filenames:
        return "excluded_filename"

    if record.extension in config.excluded_extensions:
        return "binary_or_non_text"

    if not record.extension:
        return "no_extension"

    for pattern in config.excluded_patterns:
        if pattern.search(record.path.name):
            return "pattern_match"

    if config.max_file_size and record.size > config.max_file_size:
        return "file_too_large"

    if config.max_line_length and record.size > config.max_line_length:
        try:
            with open(record.path, errors="ignore") as fh:
                for line in fh:
                    if len(line) > config.max_line_length:
                        return "long_line_minified_or_data"
        except OSError:
            pass

    if config.included_extensions and record.extension not in config.included_extensions:
        return "extension_not_in_whitelist"

    return ""
