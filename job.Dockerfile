FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers

# Kalisio dev tooling (k-clone, k-dev, k-pull, ...): DEVELOPMENT_DIR is
# expected to be bind-mounted at /workspace at `docker run` time (see
# the knowledge window in services.yaml), same layout as on a dev workstation.
ENV DEVELOPMENT_DIR=/workspace
ENV KALISIO_DEVELOPMENT_DIR=/workspace/kalisio
ENV PATH="${PATH}:/workspace/kalisio/development/scripts"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
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

RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean --all --yes

CMD ["micromamba", "run", "-n", "base", "python", "-m", "ingestion.main"]
