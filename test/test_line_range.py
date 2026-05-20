"""Tests for chunk line-range computation."""

from __future__ import annotations

from ingestion_job.chunking._line_range import compute_line_range


def test_js_breadcrumb_header_stripped_and_located() -> None:
    source = (
        "// header file\n"
        "const a = 1\n"
        "\n"
        "function foo() {\n"
        "    return 42\n"
        "}\n"
    )
    chunk = "// src/foo.js :: foo\nfunction foo() {\n    return 42\n}"
    start, end = compute_line_range(chunk, source)
    assert start == 4
    assert end == 6


def test_markdown_breadcrumb_two_line_prefix_stripped() -> None:
    source = (
        "# Title\n"
        "\n"
        "## Section\n"
        "\n"
        "Paragraph one is here.\n"
        "Continued line.\n"
        "\n"
        "## Other\n"
        "Other content.\n"
    )
    chunk = (
        "Context: Title > Section\n"
        "Source: docs/x.md\n"
        "\n"
        "Paragraph one is here.\n"
        "Continued line."
    )
    start, end = compute_line_range(chunk, source)
    assert start == 5
    assert end == 6


def test_vue_template_html_comment_header_stripped() -> None:
    source = (
        "<template>\n"
        "  <div class=\"foo\">\n"
        "    hello\n"
        "  </div>\n"
        "</template>\n"
    )
    chunk = "<!-- src/A.vue [template] -->\n<div class=\"foo\">\n    hello\n  </div>"
    start, end = compute_line_range(chunk, source)
    assert start == 2
    assert end == 4


def test_anchor_not_found_returns_full_file_range() -> None:
    source = "line 1\nline 2\nline 3\n"
    chunk = "// path :: x\ncompletely unrelated content not in source"
    start, end = compute_line_range(chunk, source)
    assert start == 1
    assert end == 3


def test_empty_source_returns_one_one() -> None:
    assert compute_line_range("// x\nfoo", "") == (1, 1)


def test_end_line_capped_at_total_lines() -> None:
    source = "a\nb\nc\n"
    chunk = "// path\na\nb\nc\nd\ne"
    _, end = compute_line_range(chunk, source)
    assert end == 3


def test_no_header_uses_raw_text() -> None:
    source = "alpha\nbeta\ngamma\n"
    chunk = "beta\ngamma"
    start, end = compute_line_range(chunk, source)
    assert start == 2
    assert end == 3
