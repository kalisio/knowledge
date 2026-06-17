# nb02 · Markdown chunking

**Notebook:** `notebooks/nb02_markdown_chunking_strategies.ipynb` · **Status:** done

## The question

How should Markdown (the bulk of the KDK / Kano docs) be split into chunks so that retrieval
is accurate **and** code blocks are never cut in half?

## Strategies compared

Several strategies were benchmarked against a recursive-character-text-split (RCT) baseline,
including AST-aware variants that merge small sections and prepend a heading **breadcrumb**
to each chunk for context.

<!-- TODO: from the notebook — the full strategy list (A–E) and a one-line description each. -->

| Strategy | Idea | Result |
| --- | --- | --- |
| RCT (baseline) | Fixed-size recursive character split | reference point |
| AST-merge + breadcrumb (winner) | Split on Markdown AST, merge small nodes, prefix heading breadcrumb | see below |

## Result

**Winner: AST-merge + breadcrumb, `chunk_size = 500`.**

- **100% code-block integrity** — no code block split across chunks.
- **~2× the bundle efficiency** of the RCT baseline (more relevant context per token).

Measured with the shared deterministic retrieval metrics (`recall@k`, source coverage) over
a gold query set.

## Reproduce

The benchmark is scriptable (gold set → deterministic metrics → parameter sweep on the
winner). See the `chunking_lab/` helpers next to the notebook for the exact entry points.

## In production

This strategy is the Markdown chunker in the [ingestion pipeline](/architecture/ingestion-pipeline).
