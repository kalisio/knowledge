# Knowledge

Internal starter project for RAG experiments on Kalisio documentation.

This repository currently provides a simple local pipeline:

1. chunk Markdown documentation
2. generate embeddings
3. ingest vectors into Qdrant
4. query Qdrant
5. optionally ask an LLM through Ollama with retrieved context

## Prerequisites

- Miniconda or Anaconda
- Docker and Docker Compose
- Optional: Ollama, either local or on a machine reachable on the LAN

## Install Conda

Install Miniconda with the official installer:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda_installer.sh
bash /tmp/miniconda_installer.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
```

Reload your shell:

```bash
source ~/.bashrc
```

Check installation:

```bash
conda --version
```

## Install Project Dependencies

The project uses Python 3.11, managed through `environment.yml`.

Create the conda environment and install all dependencies:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate knowledge
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
python scripts/chunk_md.py
```

Output:

```text
outputs/md_chunks.jsonl
```

### 2. Generate embeddings

```bash
python scripts/embed_chunks.py
```

Output:

```text
outputs/md_chunks_with_embeddings.jsonl
```

### 3. Ingest into Qdrant

```bash
python scripts/ingest_chunks_to_qdrant.py
```

### 4. Query Qdrant directly

```bash
python scripts/query_qdrant.py
```

### 5. Ask an LLM with retrieved context

```bash
python scripts/ask_llm_rag.py
```

## LLM Configuration

The RAG script supports two LLM backends: **Ollama** (default) and **Anthropic Claude**.

Configuration is done through the `.env` file at the project root:

```env
# LLM provider: "ollama" (default) or "anthropic"
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://192.168.1.109:11434
OLLAMA_MODEL=qwen2.5:7b

# Only needed when LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama (default)

#### Ollama on the same machine

Set in `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

#### Ollama on another machine in the local network

Set in `.env`:

```env
OLLAMA_BASE_URL=http://<LAN_IP>:11434
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

### Anthropic Claude

Set in `.env`:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Token usage and cost are tracked automatically in `outputs/token_ledger.json` when using Claude. Ollama calls are not tracked (free local model).

## Notes

- Activate the conda environment with `conda activate knowledge` before running scripts.
- Generated files under `outputs/` are local artifacts and should not be treated as source files.
- `qdrant_data/` is local runtime data and should not be committed.
