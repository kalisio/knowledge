# nb03 · JS & Vue chunking

**Notebook:** `notebooks/nb03_js_vue_chunking.ipynb` · **Status:** done

## The question

Source code is not prose. How should **JavaScript** and **Vue single-file components (SFCs)**
be chunked so that functions, components and their context stay coherent?

## Approach

- **JavaScript** — recursive, structure-aware splitting with a breadcrumb prefix (so a chunk
  carries where it sits in the file).
- **Vue SFC** — a dispatcher that handles the `<template>` / `<script>` / `<style>` blocks of
  an SFC separately, with an expanded breadcrumb. Reached **~97% accuracy** on the Vue split
  tests.
- **Hybrid retrieval** — added **BM25 + Reciprocal Rank Fusion (RRF)** on top of dense
  retrieval, combining lexical and semantic matches.

<!-- TODO: from the notebook — what the Vue "split tests" measure, and the BM25/RRF gain
     numbers vs dense-only. -->

## In production

The JS and Vue chunkers feed the [ingestion pipeline](/architecture/ingestion-pipeline);
the hybrid (BM25 + RRF) retrieval informs [RAG: indexing & retrieval](/architecture/rag).
