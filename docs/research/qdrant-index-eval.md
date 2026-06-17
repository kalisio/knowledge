# nb06 · Qdrant index & evaluation

**Notebook:** `notebooks/nb06_qdrant_index_and_eval.ipynb` · **Status:** 🚧 in progress

## The question

The earlier notebooks evaluated strategies in isolation. Do the choices still hold **at
production scale**, inside the real vector store — for dense, hybrid and reranked retrieval?

## Approach

- Build a **reproducible Qdrant index** from the selected corpus chunks (the winners from
  nb02–nb04, embedded with the nb05 model).
- **Re-run the nb05 evaluation against Qdrant** rather than an in-memory index, validating
  dense retrieval, hybrid (BM25 + RRF) retrieval and reranking at scale.

<!-- TODO: this notebook is in progress — fill in the production-scale numbers (dense vs
     hybrid vs reranked) once finalised. -->

## In production

Confirms the choice of Qdrant and the end-to-end [RAG: indexing & retrieval](/architecture/rag) path.

::: tip Status
This notebook is still in progress; treat its numbers as provisional until marked done in the
[overview](/research/introduction).
:::
