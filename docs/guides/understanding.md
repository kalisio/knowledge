# Understanding knowledge

AI is becoming essential to our line of work increasingly capable, but also increasingly expensive. The goal of **knowledge** is to give Kalisio developers an AI assistant that is both context-aware and token-efficient, regardless of which coding agent they use.

## Why not an existing solution

Several open-source projects already exist to index code and reduce token usage by improving context, such as [claude-context](https://github.com/zilliztech/claude-context). But none of them fit our needs:

![guide why-not](../images/guide-why-not.png)

- **Paid dependency** : solutions like claude-context require an OpenAI API key for embeddings, which goes against our open-source philosophy.
- **Too agent-specific** : most are built around a single coding agent. The field moves fast enough that today's dominant tool may not exist tomorrow; we don't want to lock ourselves in.
- **Too generic** : they treat all code the same. We want chunking and handlers tuned to our actual stack, `.vue`, `.js`, `.json`.

## What knowledge provides

knowledge gives every coding agent the same structured context, through a ready-to-use [configuration per agent](./agent-config/claude-code) and three complementary tools:

![guide provides](../images/guide-provides.png)

- **[Semantic code search](./semantic-search.md)** : the Kalisio codebase is chunked and embedded into [Qdrant](https://qdrant.tech/).
- **[Git intelligence](./git-intelligence.md)** : commit history is parsed into hotspot scores, co-change patterns & bus-factor risk.
- **[Dependency graph](./dependency-graph.md)** : static AST analysis maps every import to know what depends on what.
