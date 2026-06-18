# nb02 · Markdown chunking

**Notebook:** `notebooks/nb02_markdown_chunking_strategies.ipynb` · **Status:** done

## The question

Markdown is the bulk of the docs. How should it be split so a **code-generation agent** gets
enough, intact context? Three design principles drove everything:

- **Code-block integrity** — a truncated code example is worse than a missing one: the agent
  receives an incomplete snippet and completes it incorrectly.
- **Heading context** — the position of a chunk in the `h1 > h2 > h3` hierarchy helps the agent
  produce correct imports and API calls.
- **Context sufficiency over raw relevance** — the relevant criterion is whether a retrieved chunk
  contains enough (signature + parameters + example) to generate correct code, not topical
  similarity alone.

## Step 1 — Splitter panorama

Five LangChain Markdown splitters were demonstrated on a representative API page and scored on
four axes:

| Splitter | Principle | Controls size | Preserves structure | Code integrity | Heading metadata |
|---|---|:--:|:--:|:--:|:--:|
| RCT | character-based recursive | yes | no | no | no |
| MHS | split on headings | no | yes (sections) | depends on section size | yes |
| MTS | Markdown-aware separators | yes | partial | partial | no |
| EMSTS | Markdown AST parsing | no (atoms) | **yes (perfect)** | **yes (perfect)** | yes (rich) |

No single splitter satisfies all four criteria. EMSTS never breaks a code block or table but
emits one atom per element (too granular for retrieval). The selected approach combines **EMSTS
parsing + merging adjacent atoms + a heading breadcrumb**.

## Step 2 — Four candidate strategies

| ID | Strategy | Idea |
|---|---|---|
| A | RCT baseline | recursive character split with heading-aware separators |
| B | MHS + RCT | header split → RCT re-split; heading hierarchy kept in the payload |
| C | AST-Merge | EMSTS atoms merged under the same heading up to size — code blocks never split |
| D | AST-Merge + Breadcrumb | C, plus a `Context: h1 > h2 > h3` prefix injected into the chunk text |

## Step 3 — How they were evaluated

Each strategy chunked all of `kdk/docs` into its own Qdrant collection, then was scored with
**deterministic, source-level metrics** (not an LLM-judge-per-chunk rubric):

- **recall@5 / hit@5** — are the gold source files in the Top-5?
- **code_integrity** — fraction of retrieved chunks with balanced ` ``` ` fences.
- **bundle_efficiency** — on-topic chars ÷ total chars in the concatenated Top-K bundle (the
  agent reads the whole bundle, so wasted chars are wasted context).

The query set is **439 queries auto-mined from the docs** — each a page title, section heading,
or API symbol, with its source file(s) as ground truth (112 title / 266 symbol / 61 section).
This replaced an earlier 46-question FAQ set whose ground-truth sources were empty.

## Results

| Metric | A (RCT) | B (MHS+RCT) | C (AST-Merge) | D (AST+Breadcrumb) |
|---|--:|--:|--:|--:|
| Recall@5 | 0.862 | 0.869 | 0.860 | 0.867 |
| Hit@5 | 0.902 | 0.909 | 0.893 | 0.902 |
| Code integrity | 0.984 | 0.954 | **1.000** | **1.000** |
| Bundle efficiency | 0.070 | 0.112 | 0.121 | **0.142** |

- **Recall is effectively equal** — all four are within ~1 pp (~0.86). File-level retrievability
  does not distinguish the strategies on this corpus.
- **Code integrity is a strict requirement** — character-based A/B leave 2–5% of chunks with
  unbalanced code fences; AST-based C/D reach **100%**. For a code agent, a chunk with an
  unbalanced fence is worse than a missing chunk.
- **Bundle efficiency is the distinguishing metric** — strategy D's breadcrumb approximately
  **doubles** on-topic density relative to A (0.142 vs 0.070), so approximately twice as much
  relevant documentation fits the same context budget.
- **Symbol queries are the most strategy-sensitive** — this is the query form an agent uses when
  looking up a specific API.

**Known tradeoff:** on bare-symbol lookups, A/B score ~0.857 recall vs D's 0.820 — the breadcrumb
slightly reduces symbol-level embedding similarity. In practice an agent's query is closer to
"how to use `<symbol>` in `<context>`", where the breadcrumb is beneficial rather than detrimental.

## Parameter sweep on the selected strategy

Varying `chunk_size` on strategy D:

| chunk_size | Recall@5 | Integrity | Bundle eff. |
|--:|--:|--:|--:|
| **500** | **0.877** | **1.000** | **0.147** |
| 1000 | 0.867 | 1.000 | 0.142 |
| 1500 | 0.869 | 1.000 | 0.141 |

`chunk_size = 500` scores highest on all three: smaller chunks produce finer-grained sections, so
the Top-5 bundle contains less off-topic content.

## Verdict

**Strategy D — AST-Merge + Breadcrumb, `chunk_size = 500`.** It scores highest on the metrics
relevant to code generation (100% integrity, best bundle efficiency) and is approximately equal to
the others on recall. This is the Markdown chunker in the
[ingestion pipeline](/architecture/ingestion-pipeline).

## Beyond retrieval: a code-generation check

Deterministic metrics measure retrieval quality, but the real product question is whether an agent
fed only the Top-K bundle writes correct Kalisio code. Five Kalisio coding tasks were run through
all four strategies; qualitative review found that D's breadcrumb tags (e.g.
`Context: Globe Mixins > Methods > setStyle`) provide clearer hierarchical context than A's raw
fragments, which sometimes include unrelated API surface from the same file.
