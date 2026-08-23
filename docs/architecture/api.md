# API endpoints

The FastAPI service that serves retrieval over the indexed corpus — the HTTP surface a
coding agent (or any client) queries.

## Overview

Authentication is JWT bearer (`api/auth.py`): `/health` is open, while `/ask` and `/search`
require a valid token.

<!-- TODO: where the service is deployed; configuration; how tokens are issued. -->

## Endpoints

### `GET /health`

Liveness check. No authentication.

Returns:

```json
{ "status": "ok" }
```

### `POST /ask`

Retrieval-augmented answer: retrieve the relevant chunks, then ask the configured LLM.
Requires a valid JWT.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `question` | string | yes | The natural-language question |

Returns `{ "answer": "…", "sources": [ … ], "provider": "…", "model": "…" }`. `provider` and
`model` identify the LLM that produced the answer.

### `POST /search`

Semantic search only (no LLM) — returns the top matching chunks. Requires a valid JWT.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | string | yes | The search query |
| `top_k` | integer | no | How many chunks to return (default `5`) |

Returns a bare list of chunks, best match first:

```json
[
  {
    "path": "src/store/layers.js",
    "lines": "45-78",
    "score": 0.92,
    "content": "…",
    "commit_history": ["fix: …", "feat: …"]
  }
]
```

| Field | Type | Description |
| --- | --- | --- |
| `path` | string | Repository-relative path of the file the chunk comes from |
| `lines` | string | Line range the chunk covers, e.g. `45-78` (a single number for a one-line chunk) |
| `score` | float | Cosine similarity with the query, between 0 and 1 |
| `content` | string | The chunk itself, headed by its source and symbol |
| `commit_history` | string[] | Commit subjects for that file, newest first — a sliding window (`COMMIT_HISTORY_MAX_AGE_DAYS`, 180 days) with a floor (`COMMIT_HISTORY_MIN_COMMITS`, 5) so a stable file keeps its history |
| `repo` | string | Repository the file belongs to |
| `breadcrumb` | string | Heading path, symbol or component the chunk belongs to |
| `chunk_index` | integer | Position of the chunk within its file |
