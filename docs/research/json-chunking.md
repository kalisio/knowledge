# nb04 · JSON chunking

**Notebook:** `notebooks/nb04_json_chunking.ipynb` · **Status:** done

## The question

nb02 covered Markdown, nb03 covered JS/Vue. This notebook covers the `.json` portion deferred by
nb03. Unlike JS and Vue, which are each a single kind, JSON spans several different shapes —
schemas, i18n dictionaries, package manifests, test fixtures — and each requires a different
strategy.

## The corpus

After filtering (profile `js_vue_rag`, maximum file size 200 KB; `>200` KB GeoJSON test dumps
excluded by size), 60 JSON files are retained across 5 repos:

| Category | Files | Size | Notes |
|---|--:|--:|---|
| i18n_translations | 16 | ~285 KB | translation dictionaries; 542 top-level keys, 4,464 leaves |
| schemas_validation | 31 | ~68 KB | JSON Schema; 188 properties total (median 5/file, max 24) |
| test_fixtures | 11 | ~39 KB | larger fixtures already removed by the size cap |
| package_tooling | 14 | — | project manifests |
| docs_meta / test_config / other | ≤3 each | tiny | |

The default indexed set is `schemas_validation`, `i18n_translations`, `docs_meta`, `test_config`.
`package_tooling` and `test_fixtures` are excluded by default.

## Why one strategy is not sufficient

Each category has a different natural unit:

- **Schema** — one property (field name → type + UI component + validation). The Kalisio extension
  nests `field: {component, label}`; the most common components are `KTextField` (54),
  `KSelectField` (35), `KTextareaField` (18). For form generation, one property is one retrieval
  unit.
- **i18n** — one top-level section (a component's or feature's strings). Files mix flat top-level
  labels with nested sections; maximum depth reaches 7. Splitting at every leaf would emit
  thousands of one-line chunks; splitting at the top-level key produces one chunk per section.
- **package.json** — little internal structure worth retaining; only `name` / `description` /
  `scripts` / dependency names are kept.
- **Test fixtures and others** — heterogeneous and low-signal; a generic JSON split is used as a
  fallback.

For this reason strategy C is category-aware rather than a single rule.

## Available tooling

| Tool | Behaviour |
|---|---|
| `RecursiveCharacterTextSplitter` (RCT) | Blind to JSON syntax; may split inside a string literal or between a key and its value. |
| `RecursiveJsonSplitter` | Parses the tree and descends until each sub-tree fits the size limit. Every chunk is a valid JSON sub-tree, but the name of the enclosing key is lost above the split point. |

## Four strategies

**A** RCT generic, **B** `RecursiveJsonSplitter`, **C** category-aware key split, **D** = C plus a
`// <rel_path> :: <unit>` breadcrumb header.

## Structural experiment

60 files, `chunk_size = 500`. The metric `parse_integrity` is the fraction of chunks that parse as
standalone JSON after stripping any `//` header line (analogous to nb02's `code_integrity`):

| Strategy | Chunks | Mean | Max | Oversized | Parse integrity |
|---|--:|--:|--:|--:|--:|
| A — RCT | 971 | 450 | 500 | 0.0% | 0.7% |
| B — RecursiveJsonSplitter | 946 | 380 | 499 | 0.0% | 100% |
| C — category-aware | 1,135 | 339 | 1,375 | 4.2% | 100% |
| D — category + breadcrumb | 1,135 | 388 | 1,432 | 4.6% | 100% |

- A's parse integrity is ~0.7%: a random JSON-text slice is almost never a parseable document.
  For a code agent this is the same constraint as broken code fences in nb02.
- C emits ~20% more chunks than B because each top-level property or section is its own unit,
  whereas B merges small sub-trees.
- ~4% of C/D chunks exceed `chunk_size × 1.5` (a single property holding a long `options` /
  `services` array). Splitting mid-property is treated as worse for retrieval than an oversized
  chunk.

## Retrieval experiment

Gold set: 174 queries from two routes mined from the corpus — 104 i18n value-to-key ("where is
this string set in the app?") and 70 schema property-name ("which schema defines field X with
component Y?"), across 48 files. Strategy A is excluded because §4 already showed its chunks are
unparseable.

| Strategy | hit@5 | hit@10 | MRR | i18n hit@5 | schema hit@5 |
|---|--:|--:|--:|--:|--:|
| B — RecursiveJsonSplitter | 0.920 | 0.966 | 0.703 | 0.971 | 0.843 |
| C — category-aware | 0.948 | 0.989 | 0.752 | 0.990 | 0.886 |
| D — category + breadcrumb | 0.943 | 1.000 | 0.704 | 0.981 | 0.886 |
| **F — D + BM25 (RRF)** | **0.994** | 0.994 | **0.846** | **1.000** | **0.986** |

- C scores higher than B (+2.8 pp hit@5, +4.3 pp schema hit@5). Category-aware splitting improves
  retrieval, not only chunk shape — a result the structural metrics alone could not establish.
- D is approximately equal to C under dense-only retrieval (hit@5 0.943 vs 0.948; D's MRR is
  slightly lower, 0.704 vs 0.752). The ~50-token path header slightly reduces the dense embedding
  signal on short JSON chunks, so for dense-only retrieval the breadcrumb provides no benefit.
- Hybrid (F) is the improvement: +5.1 pp hit@5 and +14 pp MRR over D. BM25 tokenizes the
  breadcrumb path (`crisis`, `schemas`, `events`, `create`) and produces a direct term match when
  a query names the component; this is the same result as nb03. Schema retrieval is harder than
  i18n under dense retrieval, because a `(property, component)` pair recurs across schemas; hybrid
  raises schema hit@5 from 0.886 to 0.986.

## Selected strategy

Strategy **D — category-aware key split + breadcrumb**, retrieved with **hybrid dense + BM25**
(matching the JS/Vue path). The breadcrumb's value here is lexical: it is realized by the BM25
component, not the dense embedding. For a dense-only deployment, C is the correct choice
(marginally higher MRR, no header overhead).

Default index set: `schemas_validation`, `i18n_translations`, `docs_meta`, `test_config` (nb01's
high and medium tiers plus the small `layers.json`). `package_tooling` remains excluded; the
chunker provides a cleaned subset (`name` / `description` / `scripts` / dependency names) for
callers that opt in. `test_fixtures` remain excluded, as their token mass is large payloads with
no API-usage signal.

This is the JSON chunker in the [ingestion pipeline](/architecture/ingestion-pipeline); the hybrid
retrieval matches [RAG: indexing & retrieval](/architecture/rag).

## Limitations

- The gold queries are mined from the same corpus the chunker runs on, so hit@K measures
  within-corpus retrieval rather than generalization. The set is small (174 queries): differences
  below ~2 pp are within sampling noise.
- Scoring used MiniLM-L6-v2 for notebook runtime. The relative ordering of strategies is robust
  across embedding models, but absolute values differ under the production model.
- i18n files are indexed per language (`core_en.json` and `core_fr.json` separately); the chunker
  does not merge translations across languages.
- A schema property is the smallest unit and is never split, so ~12% of schema chunks exceed
  `chunk_size × 1.5` when a single property holds a long array.
