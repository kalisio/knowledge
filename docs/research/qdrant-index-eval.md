# nb06 · Qdrant index & evaluation

**Notebook:** `notebooks/nb06_qdrant_index_and_eval.ipynb` · **Status:** done

## The objective

nb05 selected `Qwen3-Embedding-0.6B`. This notebook builds a real Qdrant vector store from that
model, verifies that persisted retrieval reproduces nb05's in-memory results, and compares Qdrant
against two alternative stores on identical vectors. It connects the offline benchmark to the
production retrieval pipeline: same corpus, same chunks, same embeddings, a real storage backend.

## Configuration

Collection `kalisio_qwen3_nb06_v1`, Qdrant at `localhost:6333`, model `Qwen/Qwen3-Embedding-0.6B`,
cosine distance, 1024-dim vectors.

## Corpus

The same scope as nb05: 1,251 files, 9,893 chunks.

| Repository | Chunks | | Chunk type | Chunks |
|---|--:|---|---|--:|
| kdk | 6,497 | | js (javascript) | 3,908 |
| crisis | 1,930 | | md (markdown) | 1,846 |
| kano | 648 | | vue (script) | 1,733 |
| kapp | 492 | | json (i18n) | 842 |
| skeleton | 326 | | vue (template) | 681 |
| | | | json (schemas) | 222 |

## Index build

All 9,893 chunks were encoded with Qwen3 and upserted into Qdrant (1024-dim, cosine). The
collection holds 9,893 points; encoding and building took ~117 s, with the Qdrant upsert itself at
~1,069 chunks/sec. A manifest records the index version, model, distance and counts.

## Gold-set retrieval against the persisted store

The nb05 200-question gold set was replayed against the real Qdrant collection, using the same
Qwen3 query prefix and de-duplicating chunks into file-level rankings. `hit@5`:

| Layer | EN | FR |
|---|--:|--:|
| A_symbol | 0.883 | 0.883 |
| B_docs | 0.725 | 0.688 |
| C_code | 0.733 | 0.511 |
| negative | 0.000 | 0.000 |

Overall `hit@5` is 0.778 (EN) and 0.708 (FR).

## Consistency with the nb05 in-memory benchmark

The persisted Qdrant results match nb05's in-memory Qwen3 evaluation. The only material difference
is EN `C_code` (nb05 0.756 vs Qdrant 0.733), with small `hit@1` shifts (A_symbol EN 0.500 → 0.533)
— consistent with Qdrant's approximate-nearest-neighbour index. No prefix, normalization, distance
or payload-mapping discrepancy was observed.

| Layer (EN) | nb05 in-memory | Qdrant |
|---|--:|--:|
| A_symbol | 0.883 | 0.883 |
| B_docs | 0.725 | 0.725 |
| C_code | 0.756 | 0.733 |

## Vector-store comparison

The same vectors were loaded into Chroma and LanceDB and evaluated on the same gold set. Retrieval
quality is identical across the three stores; they differ only in timing.

| Store | hit@5 | hit@10 | MRR | Query time (200 queries) |
|---|--:|--:|--:|--:|
| Qdrant | 0.743 | 0.851 | 0.511 | 6.7 s |
| Chroma | 0.743 | 0.851 | 0.511 | 8.0 s |
| LanceDB | 0.746 | 0.851 | 0.511 | 55.1 s |

Per-language `hit@5` is the same across stores (EN 0.778, FR 0.708–0.714). Build time was 1.0 s for
LanceDB and 8.1 s for Chroma (Qdrant reused its already-built collection). Qdrant has the lowest
query time in this run and reproduces the nb05 benchmark, so it remains the reference store; the
choice of store does not change retrieval quality.

## Limitations

- **Negative-query safety is weak for every store** — the `negative` layer scores 0% safety at
  `hit@5` / `hit@10`: for out-of-scope queries the retriever still returns its nearest chunks.
  This is a property of the retrieval policy, not the vector store, and is addressed in the
  reranking / abstention layer rather than by changing databases.
- The gold set is the nb05 200-question set, mined from and evaluated on the same corpus, so
  `hit@k` measures within-corpus retrieval rather than generalization.
