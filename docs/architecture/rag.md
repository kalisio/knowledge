# RAG: indexing & retrieval

## Indexing

<!-- TODO: chunk → embed → store in Qdrant. Cross-link to the ingestion pipeline. -->

```mermaid
flowchart LR
  files[source files] --> chunk[chunk] --> embed[embed] --> qdrant[(Qdrant)]
```

## Retrieval

<!-- TODO: question → embed → top-k search → raw chunks back to the agent. -->

```mermaid
flowchart LR
  q[Question] --> embq[embed query] --> search[top-k search] --> qdrant[(Qdrant)]
  qdrant --> chunks[raw chunks] --> agent[agent context]
```

## Why RAG: with vs without

| Approach | What the agent reads | Input tokens (approx.) |
| --- | --- | --- |
| Without RAG | ~30 files | ~40k |
| With RAG | 1 `/search` call → ~5 chunks | ~8k |

::: tip
See [Research](/research/introduction) for the experiments behind these figures.
:::
