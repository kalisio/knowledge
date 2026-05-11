"""Production chunking for the Kalisio code-generation RAG.

Exposes four public entry points, one per file type, plus a batch
dispatcher:

- :func:`chunk_markdown` — nb02 winner (AST-Merge + Breadcrumb)
- :func:`chunk_js`       — nb03 winner (Recursive JS + breadcrumb)
- :func:`chunk_vue`      — nb03 winner (SFC dispatcher + expanded breadcrumb)
- :func:`chunk_json`     — nb04 winner (category-aware key split + breadcrumb)
- :func:`chunk_files`    — dispatches a batch by file extension

Every function returns ``list[dict]`` with keys ``text`` and
``metadata``. ``text`` includes a breadcrumb prefix (for embedding);
``metadata`` carries the same info as structured fields (for retriever
/ reranker / UI).

Metadata schema (all file types)::

    Required:
        source       — rel_path of the original file
        strategy     — strategy tag, e.g. "D_js_breadcrumb"
        chunk_index  — 0-based position within the file
        breadcrumb   — structured dict:
                         {"path": str, "symbol": str, "block": str}
    Optional:
        block_type   — Vue: "template" / "script" / "style"
                       JSON: semantic category
                             (schemas_validation / i18n_translations / …)
        doc_title    — Markdown: first h1 title

Internal layout (added in nb04 — ``src/chunking.py`` was promoted to a
package so each file-type's strategies live in their own module and
nb04's JSON code didn't balloon a single file past 600 lines):

- ``markdown.py``       — MD strategies A/B/C/D, helpers
- ``js.py``             — JS splitter (D)
- ``vue.py``            — Vue SFC splitter (D/E)
- ``json_chunking.py``  — JSON strategies (nb04)
- ``api.py``            — ``chunk_files`` dispatcher
"""

from __future__ import annotations

from .api import chunk_files
from .js import JS_CHUNK_OVERLAP, JS_CHUNK_SIZE, JS_WINNER, chunk_js
from .json_chunking import (
    JSON_CHUNK_OVERLAP,
    JSON_CHUNK_SIZE,
    JSON_INDEXED_CATEGORIES,
    JSON_WINNER,
    chunk_json,
    json_category,
)
from .markdown import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MD_STRATEGIES,
    MD_WINNER,
    _merge_atoms,
    chunk_ast_breadcrumb,
    chunk_ast_merge,
    chunk_markdown,
    chunk_mhs_rct,
    chunk_rct,
    get_doc_title,
    heading_key,
)
from .vue import VUE_WINNER, chunk_vue

# Legacy alias — tests/nb02_sweep_winner.py still import this name.
STRATEGIES = MD_STRATEGIES
WINNER_STRATEGY = MD_WINNER

__all__ = [
    # chunkers
    "chunk_markdown",
    "chunk_js",
    "chunk_vue",
    "chunk_json",
    "chunk_files",
    # markdown strategies (exported for tests / benchmarks)
    "chunk_rct",
    "chunk_mhs_rct",
    "chunk_ast_merge",
    "chunk_ast_breadcrumb",
    "_merge_atoms",
    "get_doc_title",
    "heading_key",
    "MD_STRATEGIES",
    # winners / constants
    "MD_WINNER",
    "JS_WINNER",
    "VUE_WINNER",
    "JSON_WINNER",
    "WINNER_STRATEGY",
    "STRATEGIES",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "JS_CHUNK_SIZE",
    "JS_CHUNK_OVERLAP",
    "JSON_CHUNK_SIZE",
    "JSON_CHUNK_OVERLAP",
    # JSON helpers
    "json_category",
    "JSON_INDEXED_CATEGORIES",
]
