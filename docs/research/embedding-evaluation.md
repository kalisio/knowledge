# nb05 · Embedding evaluation

**Notebook:** `notebooks/nb05_embedding_evaluation.ipynb` · **Status:** done

## The question

Select the production embedding model for retrieval. The corpus is English-only, but a large
share of production queries arrive in French, so cross-lingual retrieval (FR query → EN corpus)
must be measured rather than assumed.

## Evaluation design

The corpus is 1,251 files / 9,893 chunks. The gold set is **200 queries**, each written in both
English and French, across four layers that test different retrieval behaviours:

| Layer | Query form | What it tests | Queries |
|---|---|---|--:|
| `A_symbol` | API / component name | exact identifier match | 60 |
| `B_docs` | natural-language question | semantic understanding | 80 |
| `C_code` | concept → file | cross-modal (NL → code) | 45 |
| `negative` | out-of-scope question | rejection | 15 |

The headline metric is `hit@5`; `hit@1`, `hit@10` and MRR are also recorded.

## Candidate models

Three dense models were retained after dropping four from earlier runs (`e5-large` /
`e5-large-instruct` — 512-token cap forces truncation; `nomic-v1.5` — English-only, fails on FR;
`jina-code` — incompatible with the current `transformers` pin):

| Model | Family | Context |
|---|---|--:|
| `qwen3-0.6b` (`Qwen/Qwen3-Embedding-0.6B`) | multilingual + code-aware | 8K |
| `arctic-l-v2` (`Snowflake/snowflake-arctic-embed-l-v2.0`) | multilingual | 8K |
| `bge-m3` (`BAAI/bge-m3`) | multilingual | 8K |

All three accept 8K tokens, so every chunk fits without truncation. All produce 1024-dim vectors.

## Retrieval configurations compared

Four configurations: each dense model alone; **BM25** (lexical baseline); **hybrid** (the leading
dense model fused with BM25 via Reciprocal Rank Fusion, K=60 — selected from K ∈ {30, 60, 90}); and
a **cross-encoder reranker** applied to the top-20 of the hybrid run, on the natural-language
layers (`B_docs`, `C_code`) only.

BM25 alone scores well on `A_symbol` (en 0.850 / fr 0.833, since identifiers are literal tokens)
but poorly elsewhere, particularly on FR (`B_docs` fr 0.100, `C_code` fr 0.089).

## Results — hit@5 per layer

| Approach | A_symbol | B_docs | C_code | negative |
|---|--:|--:|--:|--:|
| qwen3-0.6b | **0.883** | **0.706** | 0.633 | 0.000 |
| arctic-l-v2 | 0.833 | 0.625 | 0.611 | 0.133 |
| bge-m3 | 0.808 | 0.556 | 0.556 | 0.133 |
| bm25 | 0.842 | 0.250 | 0.200 | 0.167 |
| hybrid_k60 | 0.875 | 0.575 | 0.589 | 0.033 |
| hybrid_k60 + rerank | 0.875 | 0.594 | **0.656** | 0.033 |

`qwen3-0.6b` is highest on `A_symbol` and `B_docs`; the reranked hybrid is highest on `C_code`.
On the `negative` layer, `qwen3-0.6b` reaches 0.000 (it does not return out-of-scope sources),
whereas the lexical and weaker dense models retain some false positives.

## Cost

| Model | Chunks/sec | Query mean | Query p95 | Index |
|---|--:|--:|--:|--:|
| qwen3-0.6b | 90.5 | 23.2 ms | 24.3 ms | 38.6 MB |
| arctic-l-v2 | 52.7 | 10.9 ms | 11.0 ms | 38.6 MB |
| bge-m3 | 52.9 | 11.2 ms | 12.3 ms | 38.6 MB |

`qwen3-0.6b` has the highest indexing throughput; `arctic-l-v2` / `bge-m3` have lower query
latency. Index size is equal (1024-dim).

## FR-first decision

Because production traffic is French-leaning, the decisive metric is FR `hit@5` on the
natural-language layers (`B_docs` + `C_code`). A weighted score across FR `B+C`, FR `A`, EN `B+C`
and FR rejection ranks the configurations:

| Approach | Weighted score | FR B+C | FR A | EN B+C |
|---|--:|--:|--:|--:|
| qwen3-0.6b | **0.698** | **0.624** | 0.883 | 0.736 |
| arctic-l-v2 | 0.648 | 0.592 | 0.817 | 0.648 |
| hybrid_k60 + rerank | 0.644 | 0.560 | 0.867 | 0.672 |

**Selected: `qwen3-0.6b`.** It has the highest weighted score, the best FR `B+C` retrieval, and
the highest indexing throughput. `arctic-l-v2` is the closest alternative.

## Cross-lingual experiment — would French docs close the FR→EN gap?

French queries trail English on the natural-language layers (on the leader, `B_docs` FR 0.688 vs
EN 0.725; `C_code` FR 0.511 vs EN 0.756). The corpus is English-only, so every French query must
bridge the FR↔EN gap. This experiment tests whether French documentation would recover that
difference, using `kalisio/dok` — which ships parallel English (`docs/`) and human-translated
French (`docs/fr/`) trees: 24 EN files mirroring 24 FR files (~264 KB). 45 natural-language gold
queries (EN/FR pairs) were evaluated in four settings:

| Setting | Corpus | Query | hit@5 (qwen3 / arctic / bge-m3) |
|---|---|---|---|
| EN→EN | EN docs | EN | 0.911 / 0.889 / 0.867 |
| FR→EN | EN docs | FR | 0.844 / 0.867 / 0.844 |
| FR→FR | FR docs | FR | 0.889 / 0.889 / 0.911 |
| FR→bi | both | FR | 0.889 / 0.867 / 0.889 |

- The mean cross-lingual difference recovered by FR→FR is **+0.045 hit@5**; the residual gap of
  FR→FR to the EN→EN ceiling is **+0.007** (i.e. FR monolingual reaches the EN monolingual level).
- By query category, lexical-anchor queries (shared proper nouns, URLs, version numbers) score
  1.000 in every setting — there is no cross-lingual gap when the anchor token is identical across
  languages. The gap appears only on term-divergent and general-concept queries, and is small.
- Bilingual indexing (FR→bi) is within ~0.02 of FR→FR — adding both languages to one index neither
  helps nor harms materially.

The conclusion is that with parallel EN/FR content sharing anchors, the cross-lingual gap is
approximately zero — so the FR shortfall in the main evaluation is not primarily an embedding
limitation.

## Closing the FR gap — glossary vs LLM query translation

A per-query error analysis on the main FR results separates failures by whether the FR query
contains an English identifier anchor:

| FR query carries an EN identifier? | n | FR-loss rate |
|---|--:|--:|
| Yes (`composant KChart`, `activité MapActivity`) | 114 | 10.5% |
| No (concept paraphrased, no project term) | 71 | 32.4% |

The shortfall concentrates in code targets (`.vue` 36%, `.js` 27%) and the `C_code` layer. Project
identifiers such as `mailer`, `EventLog`, `MapActivity` exist only in English; a multilingual
embedding maps `envoi d'emails` ≈ `email sending` but not ≈ `mailer`, which is project naming
rather than standard English. The FR shortfall is therefore a terminology-coverage problem, not an
embedding problem.

Two query-side approaches were compared against the original FR query: a hand-curated **glossary**
(~25 FR-concept → EN-project-term entries, appended in parentheses, not substituted) and **LLM
query translation**. Results (hit@5, positive queries):

| Model | EN_orig | FR_orig | FR_glossary | Δ glossary |
|---|--:|--:|--:|--:|
| qwen3-0.6b | 0.784 | 0.708 | **0.811** | +0.103 |
| arctic-l-v2 | 0.714 | 0.665 | 0.719 | +0.054 |
| bge-m3 | 0.659 | 0.616 | 0.692 | +0.076 |

The glossary raises FR `hit@5` above the EN baseline on `qwen3-0.6b` (0.708 → 0.811 vs EN 0.784).
The gain is largest on `C_code` (FR 0.526 → 0.726, +0.200), the layer most dependent on project
identifiers. The LLM-translation path could not be measured in this run (the local Ollama endpoint
was unreachable and entries fell back to the original FR query), so its effect is not reported.

## Selected configuration

- **Embedding model: `qwen3-0.6b`** (`Qwen/Qwen3-Embedding-0.6B`).
- **FR queries: glossary augmentation** — a deterministic FR-concept → EN-project-term glossary
  appended to the query, which recovers the FR shortfall on code-identifier queries.
- `arctic-l-v2` and `bge-m3` are the multilingual baselines; `arctic-l-v2` is the closest
  alternative if a lower query latency is required.

The selected model is used at index and query time — see
[RAG: indexing & retrieval](/architecture/rag).

## Limitations

- Gold queries are authored from and evaluated on the same corpus, so `hit@k` measures
  within-corpus retrieval, not generalization. The main gold set is 200 queries; the cross-lingual
  and augmentation sets are 45 and 200.
- The LLM query-translation path was not measured (the local model endpoint was unavailable);
  only the deterministic glossary path has measured results.
- The cross-lingual experiment uses one parallel corpus (`dok`, ~264 KB of user docs); it does not
  cover API-reference or source-code content.
- The glossary is hand-curated (~25 entries) and specific to the current corpus; new
  project-specific terms require new entries.
