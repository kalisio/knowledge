# Knowledge

Internal starter project for RAG experiments on Kalisio documentation.

This repository currently provides a simple local pipeline:

1. chunk Markdown documentation
2. generate embeddings
3. ingest vectors into Qdrant
4. query Qdrant
5. optionally ask an LLM through Ollama with retrieved context

## Prerequisites

- Python 3.11
- `uv`
- Docker and Docker Compose
- Optional: Ollama, either local or on a machine reachable on the LAN

## Install `uv`

Install `uv` with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell:

```bash
source ~/.bashrc
```

Check installation:

```bash
uv --version
```

## Install Project Dependencies

The project is pinned to Python 3.11 through `pyproject.toml` and `.python-version`.

Create the virtual environment:

```bash
uv venv --python 3.11
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies from the project configuration:

```bash
uv lock
uv sync
```

## Qdrant

Start Qdrant with Docker:

```bash
docker compose pull
docker compose up -d
```

Check that Qdrant is running:

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
```

The default REST endpoint is:

```text
http://localhost:6333
```

## Project Data Layout

Current scripts expect KDK documentation under:

```text
data/kdk/docs
```

If you start from a ZIP archive, extract it under `data/` and make sure the final structure matches:

```text
data/kdk/docs/...
```

## Run the Pipeline

All commands below should be run from the project root.

### 1. Chunk Markdown files

```bash
uv run python scripts/chunk_md.py
```

Output:

```text
outputs/md_chunks.jsonl
```

### 2. Generate embeddings

```bash
uv run python scripts/embed_chunks.py
```

Output:

```text
outputs/md_chunks_with_embeddings.jsonl
```

### 3. Ingest into Qdrant

```bash
uv run python scripts/ingest_chunks_to_qdrant.py
```

### 4. Query Qdrant directly

```bash
uv run python scripts/query_qdrant.py
```

### 5. Ask through Ollama with retrieved context

```bash
uv run python scripts/ask_ollama_rag.py
```

## Ollama Configuration

The current RAG script calls an Ollama HTTP endpoint.

### Local Ollama on the same machine

If Ollama runs on the same machine as this repository, use:

```text
http://localhost:11434/api/generate
```

In `scripts/ask_ollama_rag.py`, set:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
```

### Ollama on another machine in the local network

If Ollama runs on another machine, use that machine IP address:

```text
http://<LAN_IP>:11434/api/generate
```

Example:

```python
OLLAMA_URL = "http://192.168.1.109:11434/api/generate"
```

On the Ollama host machine, the server must listen on the LAN interface instead of only `127.0.0.1`.

For Windows, set the user environment variable:

```text
OLLAMA_HOST=0.0.0.0:11434
```

Then fully restart Ollama.

You can test connectivity from this machine with:

```bash
curl http://<LAN_IP>:11434/api/tags
```

### Using an API instead of Ollama

This repository does not yet provide a generic external LLM client.

If you want to use an API provider instead of Ollama, the minimal changes are:

1. replace the `ask_ollama()` function in `scripts/ask_ollama_rag.py`
2. point it to the provider HTTP endpoint
3. pass the provider authentication token
4. keep the retrieval step unchanged

In practice, the retrieval side remains:

- query Qdrant
- build a context block
- send the prompt and context to the chosen LLM API

## Notes

- Use `uv run python ...` as the default way to execute scripts.
- Generated files under `outputs/` are local artifacts and should not be treated as source files.
- `qdrant_data/` is local runtime data and should not be committed.
