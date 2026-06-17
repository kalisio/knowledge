---
sidebarDepth: 3
---

# Introduction

<!-- TODO: 1-2 paragraphs — what knowledge is, in plain terms. -->

**knowledge** is an AI developer assistant for the Kalisio ecosystem. It indexes the
entire codebase into a vector database and serves it to coding agents through a retrieval
API, so that agents retrieve only the chunks relevant to a task instead of reading whole files.

## The problem

<!-- TODO: why feeding the whole repo into an agent's context does not scale.
     Cost / token budget / signal-to-noise. See architecture/rag for the figures. -->

## What knowledge provides

<!-- TODO: turn each capability into a sentence; link to the architecture page that details it. -->

- **Semantic code search** — retrieval over an indexed vector database.
  See [RAG: indexing & retrieval](/architecture/rag).
- **Structure-aware ingestion** — per–file-type chunking (Markdown, JS, Vue, JSON), with
  incremental re-indexing of only what changed. See [Ingestion pipeline](/architecture/ingestion-pipeline).
- **Commit-history enrichment** — each chunk carries its file's recent commit history, so
  answers reflect how a file changed. See [Ingestion pipeline](/architecture/ingestion-pipeline).
- **Retrieval API** — `/ask` (RAG answer) and `/search` (chunks), JWT-secured.
  See [API endpoints](/architecture/api).

## How it fits the Kalisio ecosystem

<!-- TODO: which repos are indexed, how it is deployed (CI / dev machine / k8s),
     who consumes it. -->

## Next steps

- Curious about the internals? Start with the [Architecture overview](/architecture/introduction).
- Want the research behind the design? See [Research](/research/introduction).
