"""Collect structural statistics for the JS/Vue corpus under data/.

Uses ``src.corpus_filter`` for file selection so the same exclusion rules
(minified bundles, oversized data files, etc.) apply everywhere. Each
source file is read exactly once per run; all stats are derived from that
single read.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

_HELPER = Path(__file__).resolve().parents[1] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = _HELPER.parents[2]  # repo root; same regardless of file depth

from corpus_filter import scan_corpus, ScanResult  # noqa: E402


# ── categorization ──────────────────────────────────────────

CONFIG_BASENAMES = {
    "vite.config.js",
    "vitest.config.js",
    "vitest.browser.config.js",
    "webpack.config.js",
    "rollup.config.js",
}


def categorize(rel_path: str) -> str:
    """Classify a file by semantic role for retrieval-aware analysis."""
    s = rel_path.lower()
    name = Path(rel_path).name.lower()
    if ".test." in name or "/test/" in s or "/tests/" in s or "__screenshots__" in s:
        return "tests_and_fixtures"
    if "/docs/" in s:
        return "documentation"
    if "/scripts/" in s or "/tools/" in s or "/extras/tests/" in s:
        return "tooling_and_ops"
    if "/schemas/" in s:
        return "schemas_and_contracts"
    if "/services/" in s:
        return "service_logic"
    if "/router/" in s:
        return "routing"
    if "/tours/" in s:
        return "product_tours"
    if "/composables/" in s:
        return "composables"
    if "/mixins/" in s:
        return "mixins"
    if "/config/" in s or name.endswith(".config.js") or name in CONFIG_BASENAMES:
        return "config"
    if "/components/" in s:
        return "ui_components"
    if "/api/src/" in s:
        return "backend_runtime"
    if "/src/" in s:
        return "runtime_other"
    return "other"


def categorize_legacy(rel_path: str) -> str:
    """Backwards-compatible coarse category used by old notebooks."""
    detailed = categorize(rel_path)
    if detailed == "tests_and_fixtures":
        return "test"
    if detailed in {"config", "schemas_and_contracts"}:
        return "config"
    if detailed == "tooling_and_ops":
        return "build"
    if detailed == "documentation":
        return "docs"
    return "source"


# ── file + stats ────────────────────────────────────────────

@dataclass
class _FileView:
    rel_path: str
    size: int
    text: str
    lines: int
    category: str


def _load(records) -> list[_FileView]:
    views: list[_FileView] = []
    for r in records:
        try:
            text = r.path.read_text(errors="ignore")
        except OSError:
            continue
        views.append(_FileView(
            rel_path=r.rel_path, size=r.size, text=text,
            lines=text.count("\n") + 1, category=categorize(r.rel_path),
        ))
    return views


def describe(views: list[_FileView]) -> dict:
    sizes = [v.size for v in views]
    lines = [v.lines for v in views]
    by_cat: dict[str, int] = {}
    by_semantic: dict[str, int] = {}
    for v in views:
        by_semantic[v.category] = by_semantic.get(v.category, 0) + 1
        legacy = categorize_legacy(v.rel_path)
        by_cat[legacy] = by_cat.get(legacy, 0) + 1
    return {
        "count": len(views),
        "total_kb": sum(sizes) // 1024,
        "mean_bytes": int(mean(sizes)) if sizes else 0,
        "median_bytes": int(median(sizes)) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
        "mean_lines": int(mean(lines)) if lines else 0,
        "median_lines": int(median(lines)) if lines else 0,
        "max_lines": max(lines) if lines else 0,
        "by_category": dict(sorted(by_cat.items())),
        "by_semantic_category": dict(sorted(by_semantic.items())),
    }


# ── Vue-specific profiles ───────────────────────────────────

VUE_BLOCK_RE = re.compile(r"<(template|script|style)\b([^>]*)>", re.IGNORECASE)
SCRIPT_SETUP_RE = re.compile(r"<script\b[^>]*\bsetup\b", re.IGNORECASE)
EXPORT_DEFAULT_OBJ_RE = re.compile(r"export\s+default\s*\{")
MIXIN_RE = re.compile(r"\bmixins\s*:\s*\[")
# Tight composable match: identifier starting with `use` followed by PascalCase,
# invoked as a call. Avoids false positives on e.g. `useCase.foo`.
COMPOSABLE_CALL_RE = re.compile(r"\buse[A-Z]\w*\s*\(")


def vue_block_profile(views: list[_FileView]) -> dict:
    counts = {"template": 0, "script": 0, "style": 0, "script_setup": 0}
    style_multi = 0
    template_non_html = 0
    for v in views:
        seen_style = 0
        for m in VUE_BLOCK_RE.finditer(v.text):
            kind = m.group(1).lower()
            attrs = m.group(2).lower()
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "style":
                seen_style += 1
            if kind == "template" and "lang=" in attrs and "lang=\"html\"" not in attrs:
                template_non_html += 1
        if seen_style > 1:
            style_multi += 1
        if SCRIPT_SETUP_RE.search(v.text):
            counts["script_setup"] += 1
    counts["files_with_multiple_style_blocks"] = style_multi
    counts["files_with_non_html_template"] = template_non_html
    return counts


def vue_api_style(views: list[_FileView]) -> dict:
    """Classify SFC API style. Not mutually exclusive — a file can expose
    both a `<script setup>` block and a classic `<script>` with
    `export default {}`. We count each independently and report overlap."""
    has_setup = has_options = both = 0
    uses_mixins = uses_composables = 0
    for v in views:
        setup = bool(SCRIPT_SETUP_RE.search(v.text))
        options = bool(EXPORT_DEFAULT_OBJ_RE.search(v.text))
        if setup:
            has_setup += 1
        if options:
            has_options += 1
        if setup and options:
            both += 1
        if MIXIN_RE.search(v.text):
            uses_mixins += 1
        if COMPOSABLE_CALL_RE.search(v.text):
            uses_composables += 1
    return {
        "composition_api": has_setup,
        "options_api": has_options,
        "both_blocks": both,
        "uses_mixins": uses_mixins,
        "uses_composables": uses_composables,
    }


# ── public entry ────────────────────────────────────────────

def collect(scan: ScanResult | None = None) -> dict:
    result = scan or scan_corpus(ROOT / "data", profile="js_vue_rag")
    js = _load(result.included_with_extensions({".js"}))
    vue = _load(result.included_with_extensions({".vue"}))
    return {
        "js": describe(js),
        "vue": {
            **describe(vue),
            "blocks": vue_block_profile(vue),
            "api_style": vue_api_style(vue),
        },
        "_filter": {
            "profile": result.profile_name,
            "total_scanned": result.total,
            "included": len(result.included),
            "excluded": len(result.excluded),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2))
