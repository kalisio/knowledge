from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corpus_filter import FilterConfig, scan_corpus  # noqa: E402


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_js_vue_rag_profile_is_stricter_than_default(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "service.js", "export const x = 1\n")
    _write(tmp_path / "docs" / "guide.md", "# guide\n")
    _write(tmp_path / "tools" / "script.js", "console.log('x')\n")
    _write(tmp_path / "public" / "asset.js", "window.a = 1\n")
    _write(tmp_path / "test" / "feature.test.js", "describe('a', () => {})\n")

    default_scan = scan_corpus(tmp_path, profile="default")
    rag_scan = scan_corpus(tmp_path, profile="js_vue_rag")

    default_paths = {record.rel_path for record in default_scan.included}
    rag_paths = {record.rel_path for record in rag_scan.included}

    assert "docs/guide.md" in default_paths
    assert "docs/guide.md" not in rag_paths
    assert "tools/script.js" in default_paths
    assert "tools/script.js" not in rag_paths
    assert "public/asset.js" in default_paths
    assert "public/asset.js" not in rag_paths
    assert "src/service.js" in rag_paths
    assert "test/feature.test.js" in rag_paths


def test_unknown_profile_raises(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "a.js", "export {}\n")
    with pytest.raises(ValueError, match="Unknown corpus filter profile"):
        scan_corpus(tmp_path, profile="not_a_profile")


def test_explicit_config_overrides_profile(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "notes.txt", "hello\n")
    _write(tmp_path / "src" / "app.js", "console.log('hi')\n")

    cfg = FilterConfig(
        excluded_dirs=set(),
        excluded_extensions=set(),
        excluded_filenames=set(),
        excluded_patterns=[],
        max_file_size=0,
        max_line_length=0,
        included_extensions={".txt"},
    )
    scan = scan_corpus(tmp_path, profile="js_vue_rag", config=cfg)
    included = {record.rel_path for record in scan.included}

    assert "docs/notes.txt" in included
    assert "src/app.js" not in included
    assert scan.profile_name == "js_vue_rag+custom_config"


def test_exclusion_reason_pattern_match(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "vendor.min.js", "var a=1;\n")
    scan = scan_corpus(tmp_path, profile="default")
    reasons = {record.rel_path: record.exclude_reason for record in scan.excluded}
    assert reasons["src/vendor.min.js"] == "pattern_match"


@pytest.mark.parametrize("repo_name", ["kdk", "crisis", "kapp", "skeleton"])
def test_profile_regression_on_known_repos(repo_name: str) -> None:
    repo_root = ROOT / "data" / repo_name
    if not repo_root.exists():
        pytest.skip(f"repo not available: {repo_name}")

    default_scan = scan_corpus(repo_root, profile="default")
    rag_scan = scan_corpus(repo_root, profile="js_vue_rag")

    assert rag_scan.profile_name == "js_vue_rag"
    assert len(rag_scan.included) <= len(default_scan.included)

    rag_paths = {record.rel_path for record in rag_scan.included}
    src_service_candidates = sorted(repo_root.rglob("src/services/*.js"))
    if src_service_candidates:
        rel = str(src_service_candidates[0].relative_to(repo_root))
        assert rel in rag_paths

    docs_candidates = sorted(repo_root.rglob("docs/**/*.md"))
    if docs_candidates:
        rel = str(docs_candidates[0].relative_to(repo_root))
        assert rel not in rag_paths
