"""Inventory and structural stats for the ``.json`` slice of the corpus.

Deliberately *not* named ``corpus_stats.py`` — nb03 already ships
``experiments/nb03_chunking_js/corpus_stats.py`` and that module is
JS/Vue-specific (categorizes by source-tree location, has Vue block
profiles, ``_FileView`` carries line counts). JSON needs different
probes: semantic role of the file, schema property counts, i18n tree
depth, etc. The two are complementary — both are called from their
respective notebooks and both use ``src.corpus_filter`` for the scan.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

_SCHEMA_FILENAME_RE = re.compile(
    r"\.(create|update|get)(?:-[a-z]+)?\.json$", re.IGNORECASE
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corpus_filter import scan_corpus, ScanResult  # noqa: E402

DATA = ROOT / "data"


# ── semantic categorization ─────────────────────────────────

def categorize(rel_path: str) -> str:
    """Classify a .json file by what kind of content it carries.

    Distinct from ``nb03_chunking_js.corpus_stats.categorize`` — that one
    groups JS/Vue files by source-tree location (services / composables
    / mixins / …). Here we classify by JSON payload role, which is the
    dimension that determines the chunking strategy.
    """
    s = rel_path.replace("\\", "/").lower()
    name = Path(rel_path).name.lower()
    parts = s.split("/")
    if name == "package.json":
        return "package_tooling"
    if "/schemas/" in s and _SCHEMA_FILENAME_RE.search(name):
        return "schemas_validation"
    if "/i18n/" in s:
        return "i18n_translations"
    if ".vitepress" in parts and name == "manifest.json":
        return "docs_meta"
    if ".vitepress" in parts:
        return "docs_data"
    if parts[0] == "test" or "test" in parts:
        if "config" in parts:
            return "test_config"
        return "test_fixtures"
    return "other"


# Which categories carry signal for a code-generation RAG.
# Derived from nb01's priority table plus the per-category analysis
# in this notebook. ``package_tooling`` is excluded by default — the
# raw package.json is noisy and nb01 already flagged it.
INCLUDED_CATEGORIES = {
    "schemas_validation",
    "i18n_translations",
    "docs_meta",
    "test_config",
}


# ── structural probes ─────────────────────────────────────

def _max_depth(x, d: int = 0) -> int:
    if isinstance(x, dict):
        return max([_max_depth(v, d + 1) for v in x.values()] or [d])
    if isinstance(x, list):
        return max([_max_depth(v, d + 1) for v in x] or [d])
    return d


def _count_leaves(x) -> int:
    if isinstance(x, dict):
        return sum(_count_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(_count_leaves(v) for v in x)
    return 1


@dataclass
class JsonFileView:
    rel_path: str
    size: int
    text: str
    category: str
    data: object
    parse_ok: bool


def load_views(records) -> list[JsonFileView]:
    views: list[JsonFileView] = []
    for r in records:
        try:
            text = r.path.read_text(errors="ignore")
        except OSError:
            continue
        try:
            data = json.loads(text)
            parse_ok = True
        except Exception:
            data = None
            parse_ok = False
        views.append(JsonFileView(
            rel_path=r.rel_path, size=r.size, text=text,
            category=categorize(r.rel_path),
            data=data, parse_ok=parse_ok,
        ))
    return views


def _describe_sizes(sizes: list[int]) -> dict:
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "total_kb": sum(sizes) // 1024,
        "mean_bytes": int(mean(sizes)),
        "median_bytes": int(median(sizes)),
        "max_bytes": max(sizes),
    }


def _profile_schemas(views: list[JsonFileView]) -> dict:
    """JSON Schema Draft-07 + Kalisio UI extensions.

    Natural chunking unit = one top-level property.
    """
    props_per_file: list[int] = []
    field_components: dict[str, int] = {}
    has_title = has_groups = has_required = 0
    for v in views:
        if not v.parse_ok or not isinstance(v.data, dict):
            continue
        props = v.data.get("properties") or {}
        props_per_file.append(len(props))
        if v.data.get("title"):
            has_title += 1
        if v.data.get("groups"):
            has_groups += 1
        if v.data.get("required"):
            has_required += 1
        for _, pspec in props.items():
            if isinstance(pspec, dict):
                fcomp = (pspec.get("field") or {}).get("component")
                if fcomp:
                    field_components[fcomp] = field_components.get(fcomp, 0) + 1
    return {
        "files": len(views),
        "props_per_file": {
            "median": int(median(props_per_file)) if props_per_file else 0,
            "max": max(props_per_file) if props_per_file else 0,
            "total": sum(props_per_file),
        },
        "meta_flags": {
            "with_title": has_title,
            "with_groups": has_groups,
            "with_required": has_required,
        },
        "top_field_components": dict(
            sorted(field_components.items(), key=lambda kv: -kv[1])[:8]
        ),
    }


def _profile_i18n(views: list[JsonFileView]) -> dict:
    """Dict-of-{scalar, nested section}. Natural chunking unit = one top-level key."""
    top_keys_total = leaves_total = 0
    depths: list[int] = []
    nested_sections = 0
    for v in views:
        if not v.parse_ok or not isinstance(v.data, dict):
            continue
        top_keys_total += len(v.data)
        leaves_total += _count_leaves(v.data)
        depths.append(_max_depth(v.data))
        nested_sections += sum(1 for val in v.data.values() if isinstance(val, dict))
    return {
        "files": len(views),
        "total_top_keys": top_keys_total,
        "total_leaves": leaves_total,
        "nested_sections": nested_sections,
        "depth": {
            "median": int(median(depths)) if depths else 0,
            "max": max(depths) if depths else 0,
        },
    }


def _profile_package(views: list[JsonFileView]) -> dict:
    deps = scripts = 0
    for v in views:
        if not v.parse_ok or not isinstance(v.data, dict):
            continue
        deps += len(v.data.get("dependencies") or {})
        deps += len(v.data.get("devDependencies") or {})
        scripts += len(v.data.get("scripts") or {})
    return {
        "files": len(views),
        "total_dependencies": deps,
        "total_scripts": scripts,
    }


def _profile_fixtures(views: list[JsonFileView]) -> dict:
    """Wildly heterogeneous: small JSON-schemas, large GeoJSON dumps,
    observation record lists. Reported for exclusion rationale."""
    sizes = [v.size for v in views]
    kinds: dict[str, int] = {"list": 0, "dict": 0, "other": 0}
    for v in views:
        if not v.parse_ok:
            kinds["other"] += 1
        elif isinstance(v.data, list):
            kinds["list"] += 1
        elif isinstance(v.data, dict):
            kinds["dict"] += 1
        else:
            kinds["other"] += 1
    return {
        **_describe_sizes(sizes),
        "by_root_kind": kinds,
    }


# ── public entry ────────────────────────────────────────

def collect(scan: ScanResult | None = None) -> dict:
    result = scan or scan_corpus(DATA, profile="js_vue_rag")
    json_recs = result.included_with_extensions({".json"})
    views = load_views(json_recs)

    by_cat: dict[str, list[JsonFileView]] = {}
    for v in views:
        by_cat.setdefault(v.category, []).append(v)

    return {
        "_filter": {
            "profile": result.profile_name,
            "total_scanned": result.total,
            "included_total": len(result.included),
            "json_included": len(views),
        },
        "by_category": {
            cat: {
                **_describe_sizes([v.size for v in vs]),
                "included_by_default": cat in INCLUDED_CATEGORIES,
            }
            for cat, vs in sorted(
                by_cat.items(), key=lambda kv: -sum(v.size for v in kv[1])
            )
        },
        "schemas_profile": _profile_schemas(by_cat.get("schemas_validation", [])),
        "i18n_profile": _profile_i18n(by_cat.get("i18n_translations", [])),
        "package_profile": _profile_package(by_cat.get("package_tooling", [])),
        "fixtures_profile": _profile_fixtures(by_cat.get("test_fixtures", [])),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
