# Ingestion pipeline

How a Kalisio repository becomes indexed, queryable chunks.

## Pipeline stages

```mermaid
flowchart LR
  kli[kli] --> clone[clone repos] --> chunk[chunk] --> embed[embed] --> qdrant[(Qdrant)]
```

<!-- TODO: describe each stage — clone (kli), chunk (per file type), embed, store. -->

## Incremental ingestion

<!-- TODO: contrast the first full ingestion with subsequent diff-based runs. -->

```mermaid
flowchart TD
  first["First run: full index"] -.-> store[(Index)]
  next["Later runs: git diff"] --> changed[changed files only] --> rechunk[targeted re-chunk] --> store
```

## Configuration

<!-- TODO: env vars / config object fields that control ingestion (chunk size, sources…). -->

| Variable | Description | Default |
| --- | --- | --- |
| `…` | … | `…` |
