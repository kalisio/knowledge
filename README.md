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

## GPU acceleration (optional)

Embedding scripts (`experiments/early_mvp/embed_chunks.py`, `experiments/early_mvp/query_qdrant.py`, `experiments/early_mvp/ask_llm_rag.py`) and the notebooks use `sentence-transformers` with automatic device detection: CUDA if available, CPU otherwise. No code changes are needed either way — each run prints the device it picked, for example `[embed] cuda (NVIDIA GeForce RTX 3060 Ti)` or `[embed] cpu (CUDA not available)`.

### Verify your setup

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

- `True 12.x` — you're set, embedding will use CUDA.
- `False None` — CPU mode. Fine for small corpora, just slower (see benchmarks below).
- `False 13.x` (or any CUDA version newer than your NVIDIA driver supports) — your torch CUDA runtime is newer than your driver. Reinstall torch with a matching wheel. For driver 535 (CUDA ≤ 12.2):
  ```bash
  pip uninstall -y torch
  pip install --index-url https://download.pytorch.org/whl/cu121 torch
  ```

Run `nvidia-smi` to check your driver's maximum supported CUDA version.

### Rough benchmarks on the KDK corpus

| Device | Batch size | Time to embed ~2k chunks |
|---|---|---|
| RTX 3060 Ti (8 GB) | 128 | ~30 s |
| CPU (8 cores) | 32 | ~5 min |

Batch size is selected automatically by `src/embedding_utils.py` based on the active device.

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
python experiments/early_mvp/chunk_md.py
```

Output:

```text
outputs/md_chunks.jsonl
```

### 2. Generate embeddings

```bash
python experiments/early_mvp/embed_chunks.py
```

Output:

```text
outputs/md_chunks_with_embeddings.jsonl
```

### 3. Ingest into Qdrant

```bash
python experiments/early_mvp/ingest_chunks_to_qdrant.py
```

### 4. Query Qdrant directly

```bash
python experiments/early_mvp/query_qdrant.py
```

### 5. Ask an LLM with retrieved context

```bash
python experiments/early_mvp/ask_llm_rag.py
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

## Notebooks

### nb02 — Markdown Chunking Strategies

`notebooks/nb02_markdown_chunking_strategies.ipynb` compares four markdown chunking strategies for the code-generation RAG pipeline and selects the winner through deterministic, reproducible metrics (no LLM judge required).

**Strategies evaluated:**

| ID | Name | Approach |
|----|------|----------|
| A | RCT | `RecursiveCharacterTextSplitter` baseline |
| B | MHS+RCT | `MarkdownHeaderTextSplitter` → RCT re-split |
| C | AST-Merge | `ExperimentalMarkdownSyntaxTextSplitter` atoms merged per heading section |
| D | AST+Breadcrumb | AST-Merge + `Context: h1 > h2 > h3` prefix injected into chunk text |

**Winner: Strategy D (`D_ast_breadcrumb`) with `chunk_size=500`.**

D achieves 100% code block integrity and 2× the bundle efficiency of the RCT baseline while matching recall. The production implementation lives in `src/chunking.py`; downstream code imports `chunk_markdown` or `chunk_files` directly.

**Reproducing the evaluation:**

```bash
# 1. Generate the gold query set from KDK docs
python experiments/nb02_chunking_md/build_gold_query_set.py

# 2. Run deterministic retrieval metrics (requires Qdrant with ingested collections)
python experiments/nb02_chunking_md/evaluate_nb02_deterministic.py

# 3. Parameter sweep on the winner strategy
python experiments/nb02_chunking_md/nb02_sweep_winner.py

# 4. Code-generation experiment (dry-run generates context bundles)
python experiments/nb02_chunking_md/nb02_codegen_experiment.py --dry-run

# 5. Run the golden tests
pytest tests/test_chunking_golden.py -v
```

## Notes

- Activate the conda environment with `conda activate knowledge` before running scripts.
- Generated files under `outputs/` are local artifacts and should not be treated as source files.
- `qdrant_data/` is local runtime data and should not be committed.
