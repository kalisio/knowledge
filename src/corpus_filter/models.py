from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    """All filter rules in one place."""

    # Directories to skip entirely (matched by name at any depth)
    excluded_dirs: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_DIRS))

    # Extensions treated as binary / non-textual (always excluded)
    excluded_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_EXTENSIONS))

    # Exact filenames to exclude (matched case-sensitive)
    excluded_filenames: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_FILENAMES))

    # Regex patterns — any filename matching one of these is excluded
    excluded_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: [re.compile(p) for p in DEFAULT_EXCLUDED_PATTERNS]
    )

    # Files larger than this (bytes) are excluded. 0 = no limit.
    max_file_size: int = 100_000

    # Lines longer than this (chars) flag file as minified/data. 0 = no check.
    max_line_length: int = 10_000

    # Only include these extensions (whitelist). Empty = no restriction.
    included_extensions: set[str] = field(default_factory=set)


@dataclass
class FileRecord:
    """One file with its classification."""

    path: Path
    rel_path: str  # relative to scan root
    extension: str
    size: int  # bytes
    zone: str  # first directory component
    exclude_reason: str = ""  # empty if included


@dataclass
class ScanResult:
    """Holds included/excluded files with query helpers."""

    root_dir: Path
    config: FilterConfig
    profile_name: str
    included: list[FileRecord] = field(default_factory=list)
    excluded: list[FileRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.included) + len(self.excluded)

    def included_by_extension(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for record in self.included:
            groups.setdefault(record.extension or "(no ext)", []).append(record)
        return groups

    def included_by_zone(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for record in self.included:
            groups.setdefault(record.zone, []).append(record)
        return groups

    def included_with_extensions(self, exts: set[str]) -> list[FileRecord]:
        return [record for record in self.included if record.extension in exts]

    def excluded_by_reason(self) -> dict[str, list[FileRecord]]:
        groups: dict[str, list[FileRecord]] = {}
        for record in self.excluded:
            groups.setdefault(record.exclude_reason, []).append(record)
        return groups

    def summary(self) -> str:
        lines = [
            f"Corpus filter ({self.profile_name}) — {self.root_dir}",
            f"  Scanned:  {self.total:>6}",
            f"  Included: {len(self.included):>6}",
            f"  Excluded: {len(self.excluded):>6}",
            "",
            "Included by extension:",
        ]
        for ext, records in sorted(
            self.included_by_extension().items(), key=lambda item: len(item[1]), reverse=True
        ):
            size = sum(record.size for record in records)
            lines.append(f"  {ext:<12} {len(records):>5} files  {_fmt_size(size):>10}")

        lines += ["", "Included by zone:"]
        for zone, records in sorted(
            self.included_by_zone().items(), key=lambda item: len(item[1]), reverse=True
        ):
            size = sum(record.size for record in records)
            lines.append(f"  {zone:<12} {len(records):>5} files  {_fmt_size(size):>10}")

        lines += ["", "Excluded by reason:"]
        for reason, records in sorted(
            self.excluded_by_reason().items(), key=lambda item: len(item[1]), reverse=True
        ):
            size = sum(record.size for record in records)
            lines.append(f"  {reason:<28} {len(records):>5} files  {_fmt_size(size):>10}")

        return "\n".join(lines)


def _fmt_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


DEFAULT_EXCLUDED_DIRS: set[str] = {
    # Version control
    ".git",
    ".svn",
    ".hg",
    # Package managers
    "node_modules",
    "bower_components",
    ".yarn",
    ".pnpm-store",
    # Build outputs
    "dist",
    "build",
    ".output",
    ".next",
    ".nuxt",
    ".vite",
    # Coverage / profiling
    "coverage",
    ".nyc_output",
    ".c8",
    # Caches
    "__pycache__",
    ".cache",
    ".parcel-cache",
    ".turbo",
    # CI / IDE
    ".github",
    ".gitlab",
    ".vscode",
    ".idea",
}

DEFAULT_EXCLUDED_EXTENSIONS: set[str] = {
    # Images / graphics
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".ico",
    ".webp",
    ".bmp",
    ".svg",
    ".eps",
    ".ai",
    ".psd",
    # 3D models
    ".glb",
    ".gltf",
    ".obj",
    ".fbx",
    # Diagrams (binary XML)
    ".drawio",
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    # Archives
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".7z",
    ".rar",
    # Lock files
    ".lock",
    # Source maps
    ".map",
    # Compiled / binary data
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".o",
    ".a",
    ".class",
    ".jar",
    ".war",
    ".wasm",
    # Scientific / geospatial binary data
    ".dods",
    ".das",
    ".dds",
    ".nc",
    ".hdf5",
    ".h5",
    ".mbtiles",
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    # Databases
    ".db",
    ".sqlite",
    ".sqlite3",
    # Media
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    # PDF (usually scanned, not chunked as text)
    ".pdf",
    # Certificates / keys (sensitive)
    ".pem",
    ".crt",
    ".key",
    ".p12",
    ".pfx",
    # Environment / secrets
    ".env",
}

DEFAULT_EXCLUDED_FILENAMES: set[str] = {
    # Lock files
    "yarn.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "poetry.lock",
    "composer.lock",
    "Cargo.lock",
    "go.sum",
    # Changelogs (auto-generated, noisy)
    "CHANGELOG.md",
    "changelog.md",
    "CHANGES.md",
    # Licenses
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    # Package metadata
    "package.json",
    # Dotfiles with no semantic value
    ".gitignore",
    ".gitmodules",
    ".gitattributes",
    ".npmignore",
    ".eslintignore",
    ".prettierignore",
    ".dockerignore",
    # Tool configs
    ".nycrc",
    ".c8rc",
    ".eslintrc",
    ".prettierrc",
    ".babelrc",
    ".editorconfig",
    "sonar-project.properties",
    # Docker
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
}

DEFAULT_EXCLUDED_PATTERNS: list[str] = [
    r"\.min\.\w+$",
    r"\.bundle\.\w+$",
    r"\.chunk\.\w+$",
    r"-lock\.json$",
    r"^\.",
]
