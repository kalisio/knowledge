# About

**knowledge** is an AI developer assistant for the Kalisio ecosystem. It indexes the codebase into a vector database ([Qdrant](https://qdrant.tech/)), extracts git intelligence from commit history (hotspots, co-changes, bus factor), and builds a dependency graph of the codebase.

![architecture overwiew](../images/architecture-overwiew.png)

All three are served to coding agents through an MCP, so that agents retrieve only the chunks or relationships relevant to a task.
