---
sidebarDepth: 3
---

# Introduction

<!-- TODO: 1-2 paragraphs — what knowledge is, in plain terms. -->

**knowledge** is an AI developer assistant for the Kalisio ecosystem. It indexes the
entire codebase into a vector database and exposes it to coding agents through an [MCP](https://modelcontextprotocol.io/) server, so that
agents retrieve only the chunks relevant to a task instead of reading whole files.

## The problem

<!-- TODO: why feeding the whole repo into an agent's context does not scale.
     Cost / token budget / signal-to-noise. See architecture/rag for the figures. -->

## What knowledge provides

<!-- TODO: turn each capability into a sentence; link to the architecture page that details it. -->

- **Semantic code search** — retrieval over an indexed vector database.
  See [RAG: indexing & retrieval](/architecture/rag).
- **Git intelligence** — hotspots, co-change, bus factor.
  See [Git intelligence](/architecture/git-intelligence).
- **Dependency graph** — import graph and impact analysis.
  See [Dependency graph](/architecture/dependency-graph).
- **MCP tools** — the surface agents actually call.
  See [MCP tools](/architecture/mcp-tools).

## How it fits the Kalisio ecosystem

<!-- TODO: which repos are indexed, how it is deployed (CI / dev machine / k8s),
     who consumes it. -->

## Next steps

- New here? Read [Understanding knowledge](/guides/understanding).
- Want to run it? See [Getting started](/guides/getting-started).
- Curious about the internals? Start with the [Architecture overview](/architecture/introduction).
