# knowledge

[![Latest Release](https://img.shields.io/github/v/tag/kalisio/knowledge?sort=semver&label=latest)](https://github.com/kalisio/knowledge/releases)
[![CI](https://github.com/kalisio/knowledge/actions/workflows/main.yml/badge.svg)](https://github.com/kalisio/knowledge/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Retrieval-augmented question-answering over the documentation and source code of the [Kalisio platform](https://kalisio.com) and its open-source projects ([KDK](https://kalisio.github.io/kdk/), [Kano](https://kalisio.github.io/kano/) and others). Documentation and code are chunked, embedded with [sentence-transformers](https://www.sbert.net/), indexed in [Qdrant](https://qdrant.tech/), and queried through a provider-agnostic LLM client (Anthropic Claude, OpenAI, Mistral, or Kimi/Moonshot — selected from `MODEL_URL`).

## Prerequisites

- Miniconda or Anaconda
- Docker (no local image building required — images are pulled from Docker Hub)
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

The chunking, embedding, retrieval and notebook code paths use `sentence-transformers` with automatic device detection: CUDA if available, CPU otherwise. No code changes are needed either way — each run prints the device it picked, for example `[embed] cuda (NVIDIA GeForce RTX 3060 Ti)` or `[embed] cpu (CUDA not available)`. Batch size is selected automatically by [src/embedding_utils.py](src/embedding_utils.py) based on the active device.

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

## Architecture

The codebase is organised as a few independent modules under [src/](src/):

| Module | Role |
|---|---|
| [src/corpus_filter/](src/corpus_filter/) | Corpus discovery and profile-driven filtering. Public entry: `scan_corpus(profile=...)`; profiles in `profiles.py`, walker in `engine.py`. |
| [src/chunking/](src/chunking/) | Production chunkers per file type. Every function returns `list[dict]` with `text` (breadcrumb-prefixed) and `metadata` (structured fields for retriever / reranker / UI). |
| [src/embedding_utils.py](src/embedding_utils.py) | Automatic device + batch-size selection for `sentence-transformers`. |
| [src/rag_system/](src/rag_system/) | Runtime components for the v1 RAG service: `config.py` (env-driven settings), `embedding.py`, `qdrant_store.py`, `llm.py` (Ollama / Anthropic). |
| [src/api/](src/api/) | FastAPI service: `app.py` (HTTP layer), `handlers.py` (RAG pipeline), `schemas.py` (request / response models), `main.py` (entry point). |
| [src/retrieval_metrics.py](src/retrieval_metrics.py) | Deterministic retrieval metrics (recall@k, source coverage, …) shared across nb02–nb06. |

The chunking package exposes one winner per file type, all selected through the matching notebook:

| File type | Function | Strategy | Notebook |
|---|---|---|---|
| Markdown | `chunk_markdown` | AST-Merge + Breadcrumb (D) | nb02 |
| JS / `.mjs` | `chunk_js` | Recursive JS + breadcrumb | nb03 |
| Vue SFC | `chunk_vue` | SFC dispatcher + expanded breadcrumb | nb03 |
| JSON | `chunk_json` | Category-aware key split + breadcrumb | nb04 |
| any | `chunk_files` | extension dispatcher (calls the right per-type chunker) | — |

## API service

The first runnable system component is a FastAPI retriever at [api/](api/). Run it locally with:

```bash
python -m api.main
```

By default it listens on `127.0.0.1:8000` (override via `HOST` / `PORT` environment variables). Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Body `{"question": "..."}` → LLM answer plus the retrieved source chunks |

The API image is published to Docker Hub as `kalisio/knowledge-api:latest` and `kalisio/knowledge-api:${git_sha}` — see [.github/workflows/main.yml](.github/workflows/main.yml). The CI workflow needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.

### Environment

[.env.example](.env.example) lists every variable read by [src/rag_system/config.py](src/rag_system/config.py). Copy it to `.env` for local runs:

```bash
cp .env.example .env
```

## LLM Configuration

The RAG service uses a single OpenAI-compatible client for all four providers; the provider label is detected from `MODEL_URL`:

```env
MODEL_API_KEY=        # required
MODEL_NAME=           # required
MODEL_URL=            # required; must match anthropic|openai|mistral|kimi|moonshot
```

If any of the three variables is empty, or if `MODEL_URL` does not match a known provider, the API raises `RuntimeError` at startup. All four providers expose `/v1/chat/completions`, so a single `OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_URL)` instance with `model=MODEL_NAME` covers them all. Adding a new provider only requires adding one more `if "<name>" in url` branch.

| Provider | `MODEL_NAME` example | `MODEL_URL` |
|---|---|---|
| Anthropic Claude | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` |
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` |
| Mistral | `mistral-large-latest` | `https://api.mistral.ai/v1` |
| Kimi / Moonshot | `kimi-latest` | `https://api.moonshot.cn/v1` |

## Project Data Layout

The corpus is expected under `data/`:

```text
data/<repo>/<sub-tree>
```

Default scan root used by `scan_corpus` is `data/`. The KDK documentation set sits at `data/kdk/docs`. If you start from a ZIP archive, extract it under `data/` keeping the per-repo folder layout.

## Notebooks

The research history is captured in six notebooks under [notebooks/](notebooks/), with the matching helpers under `experiments/nbXX_*/`.

| # | Notebook | Topic | Status |
|---|---|---|---|
| nb01 | [nb01_corpus_discovery.ipynb](notebooks/nb01_corpus_discovery.ipynb) | Corpus discovery and filtering profiles, source for [src/corpus_filter/](src/corpus_filter/). | done |
| nb02 | [nb02_markdown_chunking_strategies.ipynb](notebooks/nb02_markdown_chunking_strategies.ipynb) | Markdown chunking strategies. Winner: `D_ast_breadcrumb` (chunk_size=500), 100% code-block integrity, 2× the bundle efficiency of the RCT baseline. Production code in [src/chunking/markdown.py](src/chunking/markdown.py). | done |
| nb03 | [nb03_js_vue_chunking.ipynb](notebooks/nb03_js_vue_chunking.ipynb) | JS and Vue SFC chunking strategies; ~97% accuracy on Vue split tests. Adds BM25 + RRF for hybrid retrieval. Production code in [src/chunking/js.py](src/chunking/js.py) and [src/chunking/vue.py](src/chunking/vue.py). | done |
| nb04 | [nb04_json_chunking.ipynb](notebooks/nb04_json_chunking.ipynb) | JSON chunking with category-aware key splitting. Production code in [src/chunking/json_chunking.py](src/chunking/json_chunking.py). | done |
| nb05 | [nb05_embedding_evaluation.ipynb](notebooks/nb05_embedding_evaluation.ipynb) | Embedding model evaluation on a 200-question bilingual (FR / EN) gold set ([outputs/nb05_gold.json](outputs/nb05_gold.json)). Recommends `Qwen/Qwen3-Embedding-0.6B`; `intfloat/multilingual-e5-large` is kept as a strong baseline. | done |
| nb06 | [nb06_qdrant_index_and_eval.ipynb](notebooks/nb06_qdrant_index_and_eval.ipynb) | Reproducible Qdrant index from the selected corpus chunks; re-runs the nb05 evaluation against Qdrant to validate dense / hybrid / reranked retrieval at production scale. | in progress |

### Reproducing the chunking benchmark (nb02)

```bash
# 1. Generate the gold query set from KDK docs
python experiments/nb02_chunking_md/build_gold_query_set.py

# 2. Run deterministic retrieval metrics (requires Qdrant with ingested collections)
python experiments/nb02_chunking_md/evaluate_nb02_deterministic.py

# 3. Parameter sweep on the winner strategy
python experiments/nb02_chunking_md/nb02_sweep_winner.py

# 4. Code-generation experiment (dry-run generates context bundles)
python experiments/nb02_chunking_md/nb02_codegen_experiment.py --dry-run
```

## Tests

Run the full suite:

```bash
pytest tests
```

Coverage:

| File | What it covers |
|---|---|
| [tests/test_chunking_golden.py](tests/test_chunking_golden.py) | End-to-end golden tests for the chunking package (marker: `golden`, requires the embedding model and the KDK docs). |
| [tests/test_corpus_filter_profiles.py](tests/test_corpus_filter_profiles.py) | Corpus filter profiles and `FilterConfig` resolution. |
| [tests/test_nb05_augmentation.py](tests/test_nb05_augmentation.py) | nb05 query augmentation helpers. |
| [tests/test_nb06_qdrant_index.py](tests/test_nb06_qdrant_index.py) | nb06 ingestion / index helpers. |
| [tests/test_system_v1.py](tests/test_system_v1.py) | System v1 FastAPI service. |

## Notes

- Activate the conda environment with `conda activate knowledge` before running scripts.
- Generated files under `outputs/` are local artifacts and should not be treated as source files.
- `qdrant_data/` is local runtime data and should not be committed.

## License

Licensed under the [MIT license](LICENSE).

Copyright (c) 2017-present [Kalisio](https://kalisio.com)

[![Kalisio](https://kalisio.github.io/kalisioscope/kalisio/kalisio-logo-black-256x84.png)](https://kalisio.com)
