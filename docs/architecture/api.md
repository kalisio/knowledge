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

Returns `{ "answer": "…", "sources": [ … ] }`.

### `POST /search`

Semantic search only (no LLM) — returns the top matching chunks. Requires a valid JWT.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | string | yes | The search query |
| `top_k` | integer | no | How many chunks to return |

Returns `{ "results": [ … ] }`.
