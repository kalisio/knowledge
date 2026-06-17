# nb05 · Embedding evaluation

**Notebook:** `notebooks/nb05_embedding_evaluation.ipynb` · **Status:** done

## The question

Which embedding model best serves a corpus that is **bilingual (French / English)** and mixes
**prose docs with source code**? The retriever is only as good as its embeddings.

## Method

- A **200-question bilingual (FR / EN) gold set** of question→expected-source pairs.
- Candidate models embedded the same corpus; retrieval scored with the shared deterministic
  metrics (`recall@k`, source coverage).
- Query-side helpers explored (augmentation, cross-lingual matching) to test robustness when
  the question language differs from the source language.

<!-- TODO: from the notebook — the full candidate list and the per-model recall@k table. -->

## Result

- **Recommended: `Qwen/Qwen3-Embedding-0.6B`.**
- **`intfloat/multilingual-e5-large`** kept as a strong baseline.

## In production

The selected model is used at index and query time — see
[RAG: indexing & retrieval](/architecture/rag).

::: warning
Switching the embedding model means re-embedding the whole corpus — vectors from different
models are not comparable.
:::
