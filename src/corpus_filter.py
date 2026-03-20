"""
Corpus filter — generic file filter for RAG ingestion pipelines.

Walks any source tree and classifies every file as "include" or "exclude"
based on configurable rules. Not tied to any specific project.

Default rules cover common noise: binaries, lock files, minified bundles,
node_modules, test fixtures, build outputs, etc. All defaults can be
overridden or extended at call time.

Usage:
    from src.corpus_filter import scan_corpus, FilterConfig

    # Use defaults
    result = scan_corpus("path/to/repo")
    print(result.summary())

    # Custom config
    cfg = FilterConfig(
        excluded_dirs=FilterConfig.DEFAULTS["excluded_dirs"] | {"vendor"},
        excluded_extensions=FilterConfig.DEFAULTS["excluded_extensions"] | {".csv"},
        max_file_size=200_000,
    )
    result = scan_corpus("path/to/repo", config=cfg)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """All filter rules in one place. Every field has a sensible default."""

    # Directories to skip entirely (matched by name at any depth)
    excluded_dirs: set[str] = field(default_factory=lambda: set(FilterConfig.DEFAULTS["excluded_dirs"]))

    # Extensions treated as binary / non-textual (always excluded)
    excluded_extensions: set[str] = field(default_factory=lambda: set(FilterConfig.DEFAULTS["excluded_extensions"]))

    # Exact filenames to exclude (matched case-sensitive)
    excluded_filenames: set[str] = field(default_factory=lambda: set(FilterConfig.DEFAULTS["excluded_filenames"]))

    # Regex patterns — any filename matching one of these is excluded
    excluded_patterns: list[re.Pattern] = field(default_factory=lambda: [
        re.compile(p) for p in FilterConfig.DEFAULTS["excluded_patterns"]
    ])

    # Files larger than this (bytes) are excluded. 0 = no limit.
    max_file_size: int = 0

    # Only include these extensions (whitelist). Empty = no restriction.
    included_extensions: set[str] = field(default_factory=set)

    DEFAULTS: dict = field(default=None, init=False, repr=False)


# Class-level defaults (not per-instance, just a reference dict)
FilterConfig.DEFAULTS = {
    "excluded_dirs": {
        # Version control
        ".git", ".svn", ".hg",
        # Package managers
        "node_modules", "bower_components", ".yarn", ".pnpm-store",
        # Build outputs
        "dist", "build", ".output", ".next", ".nuxt", ".vite",
        # Coverage / profiling
        "coverage", ".nyc_output", ".c8",
        # Caches
        "__pycache__", ".cache", ".parcel-cache", ".turbo",
        # CI / IDE
        ".github", ".gitlab", ".vscode", ".idea",
    },
    "excluded_extensions": {
        # Images / graphics
        ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff",
        ".ico", ".webp", ".bmp", ".svg",
        ".eps", ".ai", ".psd",
        # 3D models
        ".glb", ".gltf", ".obj", ".fbx",
        # Diagrams (binary XML)
        ".drawio",
        # Fonts
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
        # Archives
        ".zip", ".gz", ".tar", ".bz2", ".7z", ".rar",
        # Lock files
        ".lock",
        # Source maps
        ".map",
        # Compiled / binary data
        ".pyc", ".pyo", ".so", ".dll", ".dylib", ".o", ".a",
        ".class", ".jar", ".war",
        ".wasm",
        # Scientific / geospatial binary data
        ".dods", ".das", ".dds", ".nc", ".hdf5", ".h5",
        ".mbtiles", ".shp", ".shx", ".dbf", ".prj",
        # Databases
        ".db", ".sqlite", ".sqlite3",
        # Media
        ".mp3", ".mp4", ".wav", ".avi", ".mov",
        # PDF (usually scanned, not chunked as text)
        ".pdf",
        # Certificates / keys (sensitive)
        ".pem", ".crt", ".key", ".p12", ".pfx",
        # Environment / secrets
        ".env",
    },
    "excluded_filenames": {
        # Lock files
        "yarn.lock", "package-lock.json", "pnpm-lock.yaml",
        "Gemfile.lock", "poetry.lock", "composer.lock",
        "Cargo.lock", "go.sum",
        # Changelogs (auto-generated, noisy)
        "CHANGELOG.md", "changelog.md", "CHANGES.md",
        # Licenses
        "LICENSE", "LICENSE.md", "LICENSE.txt",
        # Package metadata
        "package.json",
        # Dotfiles with no semantic value
        ".gitignore", ".gitmodules", ".gitattributes",
        ".npmignore", ".eslintignore", ".prettierignore",
        ".dockerignore",
        # Tool configs
        ".nycrc", ".c8rc", ".eslintrc", ".prettierrc",
        ".babelrc", ".editorconfig",
        "sonar-project.properties",
        # Docker
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    },
    "excluded_patterns": [
        r"\.min\.\w+$",          # minified files: foo.min.js, bar.min.css
        r"\.bundle\.\w+$",       # bundled files
        r"\.chunk\.\w+$",        # webpack chunks
        r"-lock\.json$",         # any *-lock.json
        r"^\.",                   # dotfiles: .eslintrc.js, .prettierrc.json, etc.
    ],
}


# ────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    """One file with its classification."""
    path: Path
    rel_path: str            # relative to scan root
    extension: str
    size: int                # bytes
    zone: str                # first directory component
    exclude_reason: str = "" # empty if included


@dataclass
class ScanResult:
    """Holds all included/excluded files with query helpers."""
    root_dir: Path
    config: FilterConfig
    included: list[FileRecord] = field(default_factory=list)
    excluded: list[FileRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.included) + len(self.excluded)

    def included_by_extension(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for r in self.included:
            groups.setdefault(r.extension or "(no ext)", []).append(r)
        return groups

    def included_by_zone(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for r in self.included:
            groups.setdefault(r.zone, []).append(r)
        return groups

    def excluded_by_reason(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for r in self.excluded:
            groups.setdefault(r.exclude_reason, []).append(r)
        return groups

    def summary(self) -> str:
        lines = [
            f"Corpus filter — {self.root_dir}",
            f"  Scanned:  {self.total:>6}",
            f"  Included: {len(self.included):>6}",
            f"  Excluded: {len(self.excluded):>6}",
            "",
            "Included by extension:",
        ]
        for ext, recs in sorted(self.included_by_extension().items(),
                                 key=lambda x: len(x[1]), reverse=True):
            size = sum(r.size for r in recs)
            lines.append(f"  {ext:<12} {len(recs):>5} files  {_fmt_size(size):>10}")

        lines += ["", "Included by zone:"]
        for zone, recs in sorted(self.included_by_zone().items(),
                                  key=lambda x: len(x[1]), reverse=True):
            size = sum(r.size for r in recs)
            lines.append(f"  {zone:<12} {len(recs):>5} files  {_fmt_size(size):>10}")

        lines += ["", "Excluded by reason:"]
        for reason, recs in sorted(self.excluded_by_reason().items(),
                                    key=lambda x: len(x[1]), reverse=True):
            size = sum(r.size for r in recs)
            lines.append(f"  {reason:<28} {len(recs):>5} files  {_fmt_size(size):>10}")

        return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Core logic
# ────────────────────────────────────────────────────────────

def scan_corpus(root: str | Path, config: FilterConfig | None = None) -> ScanResult:
    """Walk *root* and classify every file as included or excluded."""
    root = Path(root).resolve()
    cfg = config or FilterConfig()
    result = ScanResult(root_dir=root, config=cfg)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in cfg.excluded_dirs]
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

            rec = FileRecord(
                path=fpath, rel_path=rel, extension=ext,
                size=size, zone=zone,
            )

            reason = _check_exclusion(rec, cfg)
            if reason:
                rec.exclude_reason = reason
                result.excluded.append(rec)
            else:
                result.included.append(rec)

    result.included.sort(key=lambda r: r.rel_path)
    result.excluded.sort(key=lambda r: r.rel_path)
    return result


def _check_exclusion(rec: FileRecord, cfg: FilterConfig) -> str:
    """Return exclusion reason, or empty string to include."""
    # 1. Exact filename
    if rec.path.name in cfg.excluded_filenames:
        return "excluded_filename"

    # 2. Binary extension
    if rec.extension in cfg.excluded_extensions:
        return "binary_or_non_text"

    # 3. No extension (usually config stubs)
    if not rec.extension:
        return "no_extension"

    # 4. Regex patterns (minified, bundled, etc.)
    for pattern in cfg.excluded_patterns:
        if pattern.search(rec.path.name):
            return "pattern_match"

    # 5. File size limit
    if cfg.max_file_size and rec.size > cfg.max_file_size:
        return "file_too_large"

    # 6. Extension whitelist (if set)
    if cfg.included_extensions and rec.extension not in cfg.included_extensions:
        return "extension_not_in_whitelist"

    return ""


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _fmt_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"
