# nb01 · Corpus discovery

**Notebook:** `notebooks/nb01_corpus_discovery.ipynb` · **Status:** done

## The question

Before anything can be chunked or embedded, we need to know **what is actually in the
corpus**: which files exist, which are worth indexing, and how each kind of content should be
handled. nb01 scans the Kalisio repositories, inventories every file, and provides the later
notebooks with a concrete per-content-type plan.

It reads sibling repos — `kdk`, `kano`, `crisis`, `kapp`, `skeleton`, `dok` — symlinked under
`data/` (`bash scripts/setup_notebook_data.sh`). KDK is used as the primary corpus throughout.

## How the corpus was explored

Eight passes: full scan → exclusion → zones → content families → documentation deep-dive →
JSON deep-dive → source-code signals → summary & decisions.

## 1. Scan and exclusion

The raw scan of KDK found **976 files (32.8 MB)** across 25 extensions. Excluding files with no
semantic value for RAG — lock files, binary images (`.png`/`.svg`/`.tif`), auto-generated
changelogs, minified data — removed **118 files (21.8 MB)**, leaving **858 files (11.0 MB)**.

By size the retained corpus consists mainly of `.js` (362 files, 1.9 MB), `.vue` (242, 835 KB),
`.json` (26, 3.5 MB) and `.md` (142, 504 KB).

## 2. Distribution by zone

| Zone | Files | Size | Top extensions |
|---|---|---|---|
| core | 305 | 921 KB | .vue 150, .js 149 |
| map | 238 | 1.3 MB | .js 142, .vue 87 |
| docs | 167 | 3.5 MB | .md 139, .xml 15, .drawio 8 |
| test | 50 | 4.6 MB | .js 21, .json 10, .ejs 9 |
| extras | 37 | 580 KB | .js 19, .mjs 17 |
| vite | 29 | 74 KB | .js 18, .vue 5 |
| (root) | 23 | 11 KB | .js 12 |
| scripts | 9 | 72 KB | .sh 6 |

## 3. Content families

Ten families emerged, each needing a different chunking approach:

| Family | Files | Size |
|---|---|---|
| code_js | 320 | 1.8 MB |
| vue_sfc | 242 | 835 KB |
| documentation_markdown | 139 | 496 KB |
| tests | 47 | 4.6 MB |
| barrel_index_js | 46 | 43 KB |
| diagrams_and_specs | 23 | 3.0 MB |
| other | 20 | 77 KB |
| json | 16 | 173 KB |
| text_outside_docs | 3 | 8 KB |
| templates | 2 | 4 KB |

## 4. Documentation deep-dive (`docs/`)

139 Markdown files, **~126,710 tokens**. The API reference accounts for the majority (116 files,
~116k tokens).

**Heading hierarchy** — 1,245 sections across the files: 137 H1, **504 H2**, **524 H3**, 80 H4.
The roughly equal H2/H3 counts are why the chunking strategy splits at `##` and re-splits at `###`.

**H2-level chunking simulation** — splitting every doc at `##` yields **642 chunks** (median 72,
mean 196 tokens). But **211 are too small** (<50 tokens → merge candidates) and **10 are too
large** (>1500 tokens → H3 re-split candidates; the largest is "Features service" at 2,981 tokens
in `api/map/services.md`). This directly motivates the merge-small / re-split-large rules in nb02.

**Special content** — tables 737, internal links 262, code blocks 184, admonitions 152, images
44, mermaid 13. Code blocks and tables must survive chunking intact, which sets nb02's
code-block-integrity requirement.

**Token budget** — ~126,710 tokens ≈ 253 chunks at 500 tokens, ≈ 126 at 1000.

## 5. JSON deep-dive

26 JSON files, 3.5 MB, but their RAG value varies widely:

| Type | Files | Size | RAG priority |
|---|---|---|---|
| schemas_validation | 8 | 18 KB | **high** |
| i18n_translations | 4 | 143 KB | medium |
| docs_meta | 1 | 624 B | medium |
| test_fixtures | 9 | 3.4 MB | low |
| test_config | 1 | 1.3 KB | low |
| package_tooling | 3 | 12 KB | exclude |

All JSON would cost ~927,649 tokens; keeping only high+medium priority costs ~41,031.
**Excluding test fixtures and tooling saves ~886,618 tokens** for almost no loss of RAG value.

## 6. Source-code signals

`core` (305 files, 504 exports) and `map` (238 files, 512 exports) contain almost all the logic.
A semantic-density scan (exports + functions + classes per file) identified the files with the
most API surface — e.g. `map/client/utils/utils.layers.js` (31 exports, 46 functions, 713 lines)
and `utils.features.js` (30 exports, 31 functions). These dense utility files require symbol-level
splitting rather than size-based splitting, which nb03 addresses.

## Decisions carried into nb02

1. **Markdown** — split at `##`, re-split oversized sections at `###`, merge tiny sections.
2. **JSON** — include schemas and i18n; exclude test fixtures and `package.json`.
3. **JS / Vue** — needs dedicated analysis (nb03); dense utility files require symbol-level splitting.
4. **Barrel files** (`index.js`) — navigation/metadata only, not primary chunks.
5. **Test files** — excluded from the initial index.

## The reusable outcome

The filtering logic was extracted into a generic corpus-filter module and run across **every**
project under `data/`, not just KDK:

| Project | Scanned | Included | Excluded |
|---|---|---|---|
| kdk | 976 | 821 | 155 |
| kano | 393 | 214 | 179 |
| crisis | 339 | 267 | 72 |
| kapp | 189 | 145 | 44 |
| skeleton | 133 | 88 | 45 |

Across all projects the main exclusion reasons were binary/non-text (364), excluded filename
(84), no extension (18) and file-too-large (15). This same module feeds the file-selection stage
of the [ingestion pipeline](/architecture/ingestion-pipeline).
