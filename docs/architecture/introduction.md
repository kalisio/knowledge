# Architecture overview

knowledge is built around three pieces that each do one job : 

- **[Ingestion job](/architecture/ingestion)**: builds and updates the knowledge base by collecting and processing the source data.
- **[API](/architecture/api)**: exposes the knowledge base through a API
- **[MCP server](/architecture/mcp)**: makes the knowledge available to coding agents through the Model Context Protocol (MCP).

![full architecture](/images/full-architecture.png)
