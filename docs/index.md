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
  - title: Semantic code search
    details: The Kalisio codebase is chunked and embedded into Qdrant
  - title: Git intelligence
    details: Commit history is parsed into hotspot scores, co-change patterns & bus-factor risk
  - title: Dependency graph
    details: Static AST analysis maps every import to know what depends on what
---

<ClientOnly>
  <home-footer />
</ClientOnly>
