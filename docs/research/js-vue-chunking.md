# nb03 · JS & Vue chunking

**Notebook:** `notebooks/nb03_js_vue_chunking.ipynb` · **Status:** done

## The question

nb02 covered Markdown. This notebook addresses the source-code portion of the corpus — `.js`
and `.vue` files — so the RAG can serve an agent that writes Kalisio code. (`.json` is handled
in nb04.) Two constraints differ from Markdown: code has no headings, so the breadcrumb approach
from nb02 must be adapted; and each Vue file contains three distinct sub-languages (template,
script, style).

## The corpus

After filtering (profile `js_vue_rag`; 1,491 files scanned, 1,107 retained; minified bundles and
oversized data files excluded):

- **JS — 566 files, ~2.0 MB.** Median 1.6 KB / 53 lines. By role: 497 source, 54 test, 15 config.
- **Vue — 351 files, ~1.25 MB.** Blocks: 585 `<template>`, 353 `<script>` (225 of them
  `<script setup>`), 65 `<style>`.

**Vue 2 / Vue 3 distribution.** 64 % of Vue files use the Composition API (`<script setup>`, with
top-level declarations), which aligns with a JS splitter. 36 % use the Options API, with
`methods: {}` / `computed: {}` nested inside `export default {}`; 102 files use mixins, so part of
their logic is not present in the file and is injected at runtime via `this`. The largest Vue
files in the corpus all use the Options API.

## Available tooling

In `langchain_text_splitters`, JS has a dedicated splitter —
`RecursiveCharacterTextSplitter.from_language(Language.JS)` (separators for `function`, `class`,
`const`, etc.). Vue has no native support: there is no `vue` entry in the `Language` enum, and no
AST-based splitter is provided (an AST splitter would require tree-sitter or `@babel/parser`). For
Vue the options are therefore a syntax-blind splitter or a purpose-built SFC dispatcher that
routes each block to the appropriate sub-splitter.

## JavaScript — strategies and retrieval

Four chunking strategies (plus a path-only variant): **A** generic recursive, **B**
`from_language(JS)`, **C** JS with a larger window (1400/200), **D** = B plus a
`// <rel_path> :: <symbol>` breadcrumb header. Structurally, JS-aware separators raise boundary
quality from 0.58 to 0.60, and the larger window halves the chunk count but reduces per-chunk
embedding signal. Structural metrics alone are not sufficient to select a strategy; the retrieval
experiment provides the basis for the decision.

**Retrieval experiment** — 191 gold queries, generated automatically from the exported symbols and
stratified by file size (66 small / 66 medium / 59 large), scored with `hit@K` (gold file in
Top-K), `sym_hit@K` (gold symbol present in the Top-K chunks) and MRR (model: MiniLM-L6-v2):

| Strategy | hit@3 | hit@5 | hit@10 | sym_hit@5 | MRR |
|---|--:|--:|--:|--:|--:|
| A — generic | 0.738 | 0.853 | 0.927 | 0.832 | 0.652 |
| B — `from_language(JS)` | 0.738 | 0.864 | 0.927 | 0.838 | 0.652 |
| C — JS, 1400/200 | 0.764 | 0.827 | 0.906 | 0.838 | 0.620 |
| D — JS + breadcrumb | 0.853 | 0.906 | 0.942 | 0.885 | 0.752 |
| **F — hybrid (D + BM25, RRF)** | **0.890** | **0.932** | **0.990** | **0.942** | **0.804** |

- The breadcrumb header (D) improves retrieval more than the choice of splitter: +11.5 pp hit@3
  and +10 pp MRR over B, at ~40 characters per chunk of overhead. The file path is the dominant
  signal; adding the nearest symbol name to the header adds a further ~1 pp.
- Hybrid retrieval (F) scores highest on every metric. Fusing the dense ranking with BM25 via
  Reciprocal Rank Fusion (`k_rrf=60`) reaches hit@10 0.990. BM25 splits identifiers on camelCase,
  so a query such as "zoom control (`getZoomControl`)" produces a direct term match; the dense
  encoder does not reliably produce this match on short utility functions.

## Vue — strategies and retrieval

The strategies mirror JS but are block-aware: **A** generic, **B** HTML-aware, **C** SFC
dispatcher (`<script>` → JS splitter, `<template>` → HTML splitter, `<style>` → its own chunk),
**D** = C plus a `<!-- <rel_path> [block] -->` breadcrumb. The SFC dispatcher (C/D) reduces the
block-mix ratio — the fraction of chunks combining HTML and JS — from 7.2 % (generic) to 0 %.

**Retrieval experiment** — 341 gold queries across three routes (258 component names, 31
composables, 52 Vue 2 registered names):

| Strategy | hit@3 | hit@5 | hit@10 | sym_hit@5 | MRR |
|---|--:|--:|--:|--:|--:|
| A — generic | 0.584 | 0.660 | 0.762 | 0.578 | 0.448 |
| B — HTML-aware | 0.589 | 0.698 | 0.774 | 0.551 | 0.484 |
| C — SFC-aware | 0.578 | 0.692 | 0.798 | 0.554 | 0.481 |
| D — SFC + breadcrumb | 0.862 | 0.924 | 0.965 | 0.877 | 0.737 |
| E — SFC + expanded breadcrumb | 0.930 | 0.959 | 0.982 | 0.889 | 0.833 |
| **F — hybrid (E + BM25, RRF)** | **0.965** | **0.979** | **0.997** | **0.950** | **0.868** |

- As with JS, the breadcrumb (D) increases hit@3 by ~27 pp over the best structural baseline.
- The expanded breadcrumb (E) addresses a tokenization mismatch: a bi-encoder tokenizes
  `KZoomControl.vue` as a single opaque sub-word sequence, so injecting the paraphrased phrase
  (`… — zoom control [template]`) provides surface tokens the query can match — hit@5 rises from
  0.924 to 0.959.
- Hybrid retrieval (F) scores highest in every category (hit@5 0.979) and improves the
  lowest-scoring category, Vue 2 registered names, where the JS splitter does not align with the
  nested `methods: {}` of the Options API.

## Shared retrieval method

Both file types converge on the same retrieval method: dense embeddings combined with BM25, fused
with Reciprocal Rank Fusion. It introduces no additional ML model (only the `rank_bm25` package
and a short fusion step) and consistently outperforms dense-only retrieval, because lexical
term-matching on identifiers covers the cases where embedding similarity between code fragments is
weakest.

## Selected strategies

| File type | Chunking | Retrieval | Result |
|---|---|---|---|
| `.js` | D — `from_language(JS)` 800/120 + breadcrumb | F — dense + BM25 (RRF) | hit@5 0.932, hit@10 0.990 |
| `.vue` | E — SFC dispatcher + expanded breadcrumb | F — dense + BM25 (RRF) | hit@5 0.979, MRR 0.868 |

Dense-only retrieval (D for JS, E for Vue) is retained as the fallback when BM25 is unavailable.
These chunkers feed the [ingestion pipeline](/architecture/ingestion-pipeline); the hybrid
retrieval informs [RAG: indexing & retrieval](/architecture/rag).

## Limitations

- The JS splitter's top-level separators do not align with the nested `methods: {}` /
  `computed: {}` of the Vue 2 Options API, so symbol-level retrieval is weakest on that category
  (E and F reduce the file-level impact). Addressing it would require an AST-based splitter
  (tree-sitter).
- Chunk sizes are character-based; they are not aligned to the embedding model's tokenizer.
- Scoring used MiniLM-L6-v2 for notebook runtime. The relative ordering of strategies is robust
  across embedding models, but absolute values differ under the production model.
