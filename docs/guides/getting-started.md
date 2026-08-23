# Getting Started

## Prerequisites

- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- A working [Kalisio development stack](https://github.com/kalisio/development)

## Install Miniconda

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

Check the installation:

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

## Environment Variables

### Shared (API + ingestion job)

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | yes | - | URL of the Qdrant instance (`http://localhost:6333`) |
| `QDRANT_COLLECTION_CODE` | yes | - | Collection for source-code chunks (ingestion) |
| `QDRANT_COLLECTION_METADATA` | yes | - | Collection for metadata chunks (ingestion) |
| `EMBEDDING_MODEL` | yes | - | HuggingFace model identifier - see tip below |
| `EMBEDDING_BATCH_SIZE` | no | `8` | Chunks embedded per batch during ingestion |
| `LOG_LEVEL` | no | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, …) |

### API service

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | yes | - | API key for the LLM provider |
| `LLM_MODEL` | yes | - | Chat model identifier (e.g. `gpt-4o-mini`) |
| `LLM_ENDPOINT` | yes | - | Base URL of any OpenAI-compatible API |
| `APP_SECRET` | yes* | - | Secret to sign and verify JWT tokens (`*` required when auth is on) |
| `KNOWLEDGE_AUTH_ENABLED` | no | `true` | Set to `false` to disable JWT auth |
| `JWT_AUDIENCE` | no | `kalisio` | Expected `aud` claim for incoming tokens |
| `JWT_ISSUER` | no | `kalisio` | Expected `iss` claim for incoming tokens |
| `JWT_ALGORITHM` | no | `HS256` | Algorithm used to verify JWT signatures |
| `TOP_K` | no | `6` | Number of chunks returned per retrieval |
| `MAX_CONTEXT_CHARS` | no | `14000` | Max characters of context fed to the LLM |
| `MAX_ANSWER_TOKENS` | no | `1024` | Max tokens the LLM may generate per answer |
| `HOST` | no | `127.0.0.1` | Host the API binds to |
| `PORT` | no | `8187` | Port the API listens on |

### Ingestion job

| Variable | Required | Default | Description |
|---|---|---|---|
| `KLI_ORGANIZATION` | no | `kalisio` | GitHub organisation passed to `k-clone` |
| `KLI_WORKSPACE` | no | `apps` | kli workspace to index |
| `GIT_HISTORY_LIMIT` | no | `10` | Recent commits attached to each chunk |

:::tip Choosing an embedding model

`EMBEDDING_MODEL` accepts any [SentenceTransformers](https://www.sbert.net/) model identifier from HuggingFace(`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`, `nomic-ai/nomic-embed-text-v1.5`)
:::

::: info OpenAI models for the LLM
`LLM_ENDPOINT` follows the [OpenAI API](https://platform.openai.com/docs/models) format, so any OpenAI-compatible provider works without code changes: OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`), [Mistral](https://mistral.ai), [Ollama](https://ollama.com) (local), and others.
:::

## Run the Project

Start the API:

```bash
python -m api.bin
```

Run the ingestion job:

```bash
python -m ingestion
```
