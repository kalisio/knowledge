# Architecture overview

knowledge is built around three pieces that each do one job : 

- **[Ingestion job](/architecture/ingestion)**: builds and updates the knowledge base by collecting and processing the source data.
- **[API](/architecture/api)**: exposes the knowledge base through a API
- **[MCP server](/architecture/mcp)**: makes the knowledge available to coding agents through the Model Context Protocol (MCP).

![full architecture](/images/full-architecture.png)

## Repository layout

The ingestion job and the API are two independent services, built into two
images and deployed apart. They share no code — not a configuration module,
not a utility package. What they do share is the Qdrant collections they
agree on: the job writes them (`ingestion/services/vectordb.py`), the API
reads them (`api/services/vectordb.py`), and the end-to-end suite runs both
against one Qdrant so a drift between the two fails a test.

Each service has the same shape: general things at the root, and folders for
the rest.

```
api/                          ingestion/
├── bin.py    entry point     ├── bin.py    entry point
├── main.py   the app         ├── main.py   the pipeline
├── config.py                 ├── config.py
├── logger.py                 ├── logger.py
├── routes.py                 ├── chunkers/   one per file type
├── schemas.py                │   ├── locator.py
└── services/                 │   ├── markdown.py, javascript.py, …
    ├── retrieval.py          └── services/
    ├── security.py               ├── scanner.py   files git tracks
    ├── llm.py                    ├── history.py   commit history
    ├── embeddings.py             ├── state.py     digests, change detection
    └── vectordb.py               ├── embeddings.py
                                  └── vectordb.py
```

Run them with `python -m api.bin` and `python -m ingestion.bin`.
