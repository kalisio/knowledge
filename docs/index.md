---
layout: home
hero:
  name: knowledge
  tagline: AI developer assistant for the Kalisio ecosystem
  image:
    src: /images/landing-knowledge.png
    alt: kalisio-knowledge
  actions:
    - theme: brand
      text: Show me more →
      link: /about/introduction
    - theme: alt
      text: Architecture
      link: /architecture/introduction
features:
- title: Context-aware AI assistant
  details: Indexes the entire Kalisio codebase into a vector database so AI agents retrieve only the relevant chunks
- title: Structure-aware, incremental ingestion
  details: Per–file-type chunking (Markdown, JS, Vue, JSON) with recent-commit-history enrichment; only changed files are re-indexed
---

<ClientOnly>
  <home-footer />
</ClientOnly>
