FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers

# Kalisio dev tooling (k-clone, k-pull, kli, ...). DEVELOPMENT_DIR is
# expected to be bind-mounted at /workspace at `docker run` time, same
# layout as on a dev workstation:
#
#   /workspace/                  <- DEVELOPMENT_DIR
#   ├── kalisio/                 <- KALISIO_DEVELOPMENT_DIR, what k-clone fills
#   │   ├── development/         <- the tooling itself (k-clone, .kalisio)
#   │   ├── kli/                 <- cloned by k-clone on first use
#   │   ├── kdk/ kano/ ...       <- the repositories to index
#   ├── irsn/  airbus/           <- the other organisations
#
# `source .kalisio` inside k-clone resolves through PATH, which is why the
# scripts directory is on it.
ENV DEVELOPMENT_DIR=/workspace
ENV DEVELOPMENT_BIN_DIR=/workspace/kalisio/development/scripts
ENV KALISIO_DEVELOPMENT_DIR=/workspace/kalisio
ENV KALISIO_DEVELOPMENT_JOBS_DIR=/workspace/kalisio
ENV IRSN_DEVELOPMENT_DIR=/workspace/irsn
ENV AIRBUS_DEVELOPMENT_DIR=/workspace/airbus
ENV PATH="${PATH}:/workspace/kalisio/development/scripts"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g yarn \
    && rm -rf /var/lib/apt/lists/*

# apt-get/npm run as root with HOME=/app, which can implicitly create /app
# owned by root; COPY --chown only chowns what it copies, not a pre-existing
# parent dir, so force ownership before copying the app code into it.
RUN mkdir -p ${HOME} && chown mambauser:mambauser ${HOME}

COPY --chown=mambauser:mambauser . ${HOME}
WORKDIR ${HOME}
USER mambauser

# The workspace is bind-mounted from the host, so its repositories belong to
# another uid. Without this git refuses to read them ("detected dubious
# ownership") and every `git ls-files` comes back empty -- the job would run
# to completion and index nothing.
RUN git config --global --add safe.directory '*'

RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean --all --yes

# KALISIO_GITHUB_URL and GITLAB_IRSN_URL are derived from the tokens, which
# are only known at run time.
# k-clone clones into the mounted workspace, so the container needs write
# access to it. The settings it reads -- KALISIO_GITHUB_URL and the rest --
# come from the service environment
# (development/workspaces/services/knowledge/knowledge.enc.env).
CMD ["micromamba", "run", "-n", "base", "python", "-m", "ingestion.bin"]
