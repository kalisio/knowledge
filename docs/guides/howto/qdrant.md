# Qdrant operations

Qdrant is the vector store the ingestion job writes to and the API reads from. The connection is
configured by two required environment variables, provisioned through the encrypted service env:

| Variable | Meaning | Example |
| --- | --- | --- |
| `QDRANT_URL` | Qdrant endpoint | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name | _(set per environment)_ |

In the commands below, `$QDRANT_URL` and `$QDRANT_COLLECTION` refer to those values.

## Run Qdrant

Qdrant runs as a Docker container. The project's tooling starts it with the `k-qdrant` script from
the Kalisio `development` workspace (`scripts/run_tests.sh` calls it), persisting data to
`./qdrant_data/`:

```bash
k-qdrant
```

`k-qdrant` is a thin wrapper over the standalone command below. The image version defaults to
`v1.18.0` (override with `QDRANT_VERSION`) and the data directory to `./qdrant_data` (override with
`QDRANT_DATA_DIR`):

```bash
docker run -d --rm --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant:v1.18.0
```

REST is on port `6333` (gRPC on `6334`). Check it is up:

```bash
curl -s http://localhost:6333/healthz
```

::: tip
Data lives in `./qdrant_data/`. Removing that directory deletes every collection — a full reset.
:::

## Web UI (dashboard)

Qdrant ships a built-in dashboard:

```
http://localhost:6333/dashboard
```

It lists collections, shows each collection's configuration (vector size, distance, point count),
lets you browse and filter points, and includes a console for running REST requests interactively.

## Inspect from the command line (REST API)

List collections:

```bash
curl -s $QDRANT_URL/collections | jq
```

Collection configuration and point count:

```bash
curl -s $QDRANT_URL/collections/$QDRANT_COLLECTION | jq
```

Exact point count:

```bash
curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/count \
  -H 'Content-Type: application/json' -d '{"exact": true}' | jq
```

List points with their payload (omit vectors for readability):

```bash
curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit": 5, "with_payload": true, "with_vector": false}' | jq
```

Fetch one point by id:

```bash
curl -s $QDRANT_URL/collections/$QDRANT_COLLECTION/points/<id> | jq
```

## Inspect from Python

Each service wraps Qdrant on its own side — `ingestion/clients/vectordb.py` writes, `api/clients/vectordb.py` reads — from `QDRANT_URL` / `QDRANT_COLLECTION_CODE`
from the runtime config:

```python
from utils import vectordb

vectordb.count()                       # indexed point count (0 if the collection is absent)
for payload in vectordb.iter_payloads():   # every stored payload, paged
    print(payload["repository"], payload["source_path"], payload["chunk_index"])
```

Or the client directly:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
client.count("<collection>", exact=True)
records, _ = client.scroll("<collection>", limit=5, with_payload=True, with_vectors=False)
```

## What is stored per point

Each point is one chunk: its embedding vector plus a payload. The payload fields (built in
`ingestion/clients/vectordb.py`):

| Field | Meaning |
| --- | --- |
| `text` | the chunk text |
| `source_path` | repo-relative file path |
| `repository` | repository name (`kdk`, `kano`, …) |
| `file_type` | file extension without the dot |
| `chunk_index` | chunk position within the file |
| `breadcrumb` | heading / symbol context |
| `commit_history` | recent commit subjects for the file |
| `text_sha1` / `file_sha1` | hashes for idempotent upsert and change detection |

Vectors use cosine distance; the vector size is the embedding model's dimension (1024 for the
selected Qwen3 model). The point id is deterministic — `uuid5(repository:source_path:chunk_index:sha1)`
— so re-indexing the same chunk overwrites its point instead of creating a duplicate.

## Verify data after ingestion

After running the ingestion job (`python -m ingestion.bin`):

1. **Count** — the point count should be greater than zero:

   ```bash
   curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/count \
     -H 'Content-Type: application/json' -d '{"exact": true}' | jq
   ```

2. **Confirm a repository is present** — scroll with a payload filter:

   ```bash
   curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/scroll \
     -H 'Content-Type: application/json' -d '{
       "filter": {"must": [{"key": "repository", "match": {"value": "kdk"}}]},
       "limit": 3, "with_payload": true, "with_vector": false
     }' | jq
   ```

3. **Confirm a specific file** — replace the filter key with `source_path`.

4. **Check retrieval end to end** — query the API and inspect the returned sources and scores.
   `/search` requires a JWT when auth is enabled (the default), so send a bearer token — or run the
   API with `KNOWLEDGE_AUTH_ENABLED=false` for a local check:

   ```bash
   curl -s -X POST http://localhost:8187/search \
     -H 'Authorization: Bearer <token>' \
     -H 'Content-Type: application/json' -d '{"query": "how to add a layer", "top_k": 5}' | jq
   ```

## Locate points with filters

Qdrant filters match on payload fields. Reusable `must` clauses:

| Target | Filter clause |
| --- | --- |
| One repository | `{"key": "repository", "match": {"value": "kano"}}` |
| One file | `{"key": "source_path", "match": {"value": "kdk/docs/api/map/services.md"}}` |
| One file type | `{"key": "file_type", "match": {"value": "vue"}}` |

## Delete

Delete points by filter — for example, drop one repository's chunks:

```bash
curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/delete \
  -H 'Content-Type: application/json' \
  -d '{"filter": {"must": [{"key": "repository", "match": {"value": "kano"}}]}}'
```

Delete specific points by id (same endpoint):

```bash
curl -s -X POST $QDRANT_URL/collections/$QDRANT_COLLECTION/points/delete \
  -H 'Content-Type: application/json' -d '{"points": ["<id>", "<id>"]}'
```

Delete the whole collection (to re-index from scratch):

```bash
curl -s -X DELETE $QDRANT_URL/collections/$QDRANT_COLLECTION
```

The ingestion job recreates the collection on its next run. In Python,
`vectordb.ensure_collection(vector_size, recreate=True)` drops and recreates it.

::: warning
Deleting by `repository` removes every chunk from that repository; re-run ingestion to restore it.
Deleting the collection (or the `qdrant_data/` volume) removes all indexed data.
:::
