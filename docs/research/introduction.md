# Research overview

This section documents the research phase of knowledge — the exploration that produced the
design decisions now used in production. The work was carried out as six Jupyter notebooks
(`nb01`–`nb06`); each page walks through one notebook: the question it asked, what was tried,
and what it concluded.

## How the experiments were evaluated

A shared, deterministic method runs across the notebooks so results are comparable:

- **Gold query sets** — curated question→expected-source pairs, including a 200-question
  bilingual (FR / EN) set, used to score retrieval objectively.
- **Deterministic retrieval metrics** — `recall@k`, source coverage and related measures,
  shared across `nb02`–`nb06` (not eyeballed).
- **Token tracking** — bundle/context token counts, to compare strategies on cost as well
  as accuracy.

## The six notebooks

| # | Topic | Outcome | Status |
|---|---|---|---|
| [nb01](/research/corpus-discovery) | Corpus discovery & filtering | Profile-driven file selection for the corpus | done |
| [nb02](/research/markdown-chunking) | Markdown chunking | Selected: AST-merge + breadcrumb, `chunk_size=500` | done |
| [nb03](/research/js-vue-chunking) | JS & Vue SFC chunking | Structure-aware chunkers + BM25/RRF hybrid retrieval | done |
| [nb04](/research/json-chunking) | JSON chunking | Category-aware key splitting | done |
| [nb05](/research/embedding-evaluation) | Embedding model evaluation | Recommends `Qwen3-Embedding-0.6B` | done |
| [nb06](/research/qdrant-index-eval) | Qdrant index & evaluation | Production-scale dense / hybrid / reranked retrieval | in progress |

## How the findings feed production

- Chunking winners (nb02–nb04) → the [ingestion pipeline](/architecture/ingestion-pipeline).
- Embedding choice (nb05) → [RAG: indexing & retrieval](/architecture/rag).
- Index & retrieval validation (nb06) → [RAG: indexing & retrieval](/architecture/rag).

::: info Notebooks & code
The notebooks and their helper code live in the repository under `docs/experiments/`
(`notebooks/nb0X_*.ipynb` plus the matching `*_lab/` helpers). They are **not** part of the
rendered site — these pages are the readable write-up of what they contain.
:::
