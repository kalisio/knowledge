"""Golden tests for src/chunking.py.

Unit tests (no external deps) verify chunk structure, metadata, and edge cases.
The golden retrieval test (marked ``golden``) embeds real KDK docs and checks
that known queries retrieve the correct source files in Top-5 via in-memory
cosine similarity — no Qdrant required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parents[2] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    STRATEGIES,
    WINNER_STRATEGY,
    _merge_atoms,
    chunk_ast_breadcrumb,
    chunk_ast_merge,
    chunk_markdown,
    chunk_mhs_rct,
    chunk_rct,
    get_doc_title,
    heading_key,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MD = """\
# Widget API

## Overview

The Widget component renders interactive cards.

## Props

### `size`

Type: `string`. Default: `"md"`.

### `color`

Type: `string`. Default: `"primary"`.

## Examples

```js
import { Widget } from '@kalisio/kdk/core.client.js'
export default {
  components: { Widget },
  setup() {
    return { size: 'lg' }
  }
}
```

## See also

- [Card](./card.md)
"""

MINIMAL_MD = "Just a paragraph, no headings at all.\n"


# ── get_doc_title ─────────────────────────────────────────────────────────────

class TestGetDocTitle:
    def test_extracts_h1(self):
        assert get_doc_title(SAMPLE_MD, "fallback") == "Widget API"

    def test_ignores_h2(self):
        assert get_doc_title("## Not an h1\ntext", "fb") == "fb"

    def test_fallback_when_no_heading(self):
        assert get_doc_title(MINIMAL_MD, "my_file") == "my_file"

    def test_first_h1_wins(self):
        md = "# First\n# Second\n"
        assert get_doc_title(md, "x") == "First"


# ── heading_key ───────────────────────────────────────────────────────────────

class TestHeadingKey:
    def test_full_hierarchy(self):
        meta = {"Header 1": "A", "Header 2": "B", "Header 3": "C"}
        assert heading_key(meta) == "A > B > C"

    def test_empty_metadata(self):
        assert heading_key({}) == "_root_"

    def test_partial(self):
        assert heading_key({"Header 1": "Top"}) == "Top"


# ── Chunk structure per strategy ──────────────────────────────────────────────

def _assert_chunk_basics(chunks: list[dict], expected_strategy: str, source: str):
    """Shared assertions every strategy must satisfy."""
    assert len(chunks) > 0, "Must produce at least one chunk"
    for i, c in enumerate(chunks):
        assert "text" in c and isinstance(c["text"], str)
        assert len(c["text"]) > 0
        assert "metadata" in c and isinstance(c["metadata"], dict)
        m = c["metadata"]
        assert m["source"] == source
        assert m["strategy"] == expected_strategy
        assert m["chunk_index"] == i
        assert "doc_title" in m


class TestChunkRCT:
    def test_structure(self):
        chunks = chunk_rct(SAMPLE_MD, "widget.md")
        _assert_chunk_basics(chunks, "A_rct", "widget.md")

    def test_respects_chunk_size(self):
        chunks = chunk_rct(SAMPLE_MD, "w.md", chunk_size=200, chunk_overlap=20)
        for c in chunks:
            # RCT may overshoot by a few chars due to separator logic, but
            # should be within a reasonable margin
            assert len(c["text"]) <= 400


class TestChunkMhsRct:
    def test_structure(self):
        chunks = chunk_mhs_rct(SAMPLE_MD, "widget.md")
        _assert_chunk_basics(chunks, "B_mhs_rct", "widget.md")

    def test_preserves_header_metadata(self):
        chunks = chunk_mhs_rct(SAMPLE_MD, "widget.md")
        has_header = any(c["metadata"].get("h1") or c["metadata"].get("h2") for c in chunks)
        assert has_header, "MHS should inject header metadata into at least one chunk"


class TestChunkAstMerge:
    def test_structure(self):
        chunks = chunk_ast_merge(SAMPLE_MD, "widget.md")
        _assert_chunk_basics(chunks, "C_ast_merge", "widget.md")

    def test_code_integrity(self):
        """AST merge must never split a code block."""
        chunks = chunk_ast_merge(SAMPLE_MD, "widget.md")
        for c in chunks:
            fence_count = c["text"].count("```")
            assert fence_count % 2 == 0, (
                f"Chunk has unbalanced fences ({fence_count}): {c['text'][:80]}…"
            )


class TestChunkAstBreadcrumb:
    def test_structure(self):
        chunks = chunk_ast_breadcrumb(SAMPLE_MD, "widget.md")
        _assert_chunk_basics(chunks, "D_ast_breadcrumb", "widget.md")

    def test_breadcrumb_prefix(self):
        chunks = chunk_ast_breadcrumb(SAMPLE_MD, "widget.md")
        has_context = any(c["text"].startswith("Context:") for c in chunks)
        assert has_context, "At least one chunk should have a Context: breadcrumb prefix"

    def test_breadcrumb_in_metadata(self):
        chunks = chunk_ast_breadcrumb(SAMPLE_MD, "widget.md")
        for c in chunks:
            assert "breadcrumb" in c["metadata"]

    def test_code_integrity(self):
        chunks = chunk_ast_breadcrumb(SAMPLE_MD, "widget.md")
        for c in chunks:
            fence_count = c["text"].count("```")
            assert fence_count % 2 == 0


# ── chunk_markdown public API ────────────────────────────────────────────────

class TestChunkMarkdown:
    def test_defaults_to_winner(self):
        chunks = chunk_markdown(SAMPLE_MD, "w.md")
        assert all(c["metadata"]["strategy"] == "D_ast_breadcrumb" for c in chunks)

    def test_explicit_strategy(self):
        for name in ("A_rct", "B_mhs_rct", "C_ast_merge", "D_ast_breadcrumb"):
            chunks = chunk_markdown(SAMPLE_MD, "w.md", strategy=name)
            assert all(c["metadata"]["strategy"] == name for c in chunks)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_markdown(SAMPLE_MD, "w.md", strategy="Z_nonexistent")


# ── _merge_atoms edge cases ──────────────────────────────────────────────────

class TestMergeAtoms:
    def test_empty_input(self):
        assert _merge_atoms([], chunk_size=500) == []

    def test_single_large_atom_kept_whole(self):
        """An atom larger than chunk_size should NOT be truncated."""
        class FakeAtom:
            def __init__(self, text, meta):
                self.page_content = text
                self.metadata = meta

        big = "x" * 1000
        atoms = [FakeAtom(big, {"Header 1": "H1"})]
        merged = _merge_atoms(atoms, chunk_size=200)
        assert len(merged) == 1
        assert len(merged[0]["text"]) == 1000


# ── Golden retrieval test ─────────────────────────────────────────────────────

GOLDEN_CASES = [
    {
        "query": "Account service",
        "expected_source": "docs/api/core/services/account.md",
        "source_type": "title",
    },
    {
        "query": "allowLocalAuthentication",
        "expected_source": "docs/api/core/hooks/hooks.users.md",
        "source_type": "symbol",
    },
    {
        "query": "`KDate`",
        "expected_source": "docs/api/core/components/time.md",
        "source_type": "section",
    },
]

KDK_DOCS = ROOT / "data" / "kdk" / "docs"


def _corpus_files(limit: int = 30) -> list[Path]:
    """Collect markdown files, ensuring golden-case files are included."""
    needed = {Path(KDK_DOCS / g["expected_source"].replace("docs/", "", 1)) for g in GOLDEN_CASES}
    pool = sorted(KDK_DOCS.rglob("*.md"))
    selected = list(needed)
    for p in pool:
        if p not in needed and len(selected) < limit:
            selected.append(p)
    return selected


@pytest.mark.golden
def test_golden_retrieval_top5():
    """Chunk real KDK docs → embed → in-memory cosine search → assert Top-5 hit."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers / numpy not installed")

    if not KDK_DOCS.is_dir():
        pytest.skip(f"KDK docs not found at {KDK_DOCS}")

    files = _corpus_files(limit=30)
    all_chunks: list[dict] = []
    for f in files:
        rel = "docs/" + str(f.relative_to(KDK_DOCS))
        text = f.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(text, source=rel, strategy=WINNER_STRATEGY))

    assert len(all_chunks) >= 20, f"Expected ≥20 chunks, got {len(all_chunks)}"

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_embeds = model.encode(chunk_texts, show_progress_bar=False, normalize_embeddings=True)

    for case in GOLDEN_CASES:
        query_embed = model.encode([case["query"]], normalize_embeddings=True)
        scores = np.dot(chunk_embeds, query_embed.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:5]
        top_sources = [all_chunks[i]["metadata"]["source"] for i in top_indices]

        assert case["expected_source"] in top_sources, (
            f"[{case['source_type']}] query={case['query']!r}: "
            f"expected {case['expected_source']!r} in Top-5, "
            f"got {top_sources}"
        )
