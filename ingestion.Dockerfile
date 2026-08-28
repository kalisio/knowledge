FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/tmp/.cache

# The embedding model lives in the image, not in a cache the workspace has to
# carry: the workspace is wiped between runs so the repositories are always
# fresh, and refetching 1.2 GB from HuggingFace nightly -- unauthenticated
# and rate-limited -- would make the job depend on a service it does not own.
ENV HF_HOME=/opt/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

# The Kalisio dev tooling is installed in the image, not in the workspace:
# the workspace is a volume that starts empty, and k-clone is what fills it.
# The job links these two directories into $KALISIO_DEVELOPMENT_DIR before
# calling k-clone, which resolves everything through it.
ENV KALISIO_TOOLING_DIR=/opt/kalisio

# The workspace the job clones into and scans, one directory per
# organisation, same layout as on a developer's machine:
#
#   /workspace/                  <- DEVELOPMENT_DIR
#   ├── kalisio/                 <- KALISIO_DEVELOPMENT_DIR, what k-clone fills
#   ├── irsn/  airbus/           <- the other organisations
ENV DEVELOPMENT_DIR=/workspace
ENV DEVELOPMENT_BIN_DIR=/opt/kalisio/development/scripts
ENV KALISIO_DEVELOPMENT_DIR=/workspace/kalisio
ENV KALISIO_DEVELOPMENT_JOBS_DIR=/workspace/kalisio
ENV IRSN_DEVELOPMENT_DIR=/workspace/irsn
ENV AIRBUS_DEVELOPMENT_DIR=/workspace/airbus

# `source .kalisio` inside k-clone resolves through PATH, which is why the
# scripts directory is on it.
ENV PATH="${PATH}:/opt/kalisio/development/scripts"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g yarn \
    && rm -rf /var/lib/apt/lists/*

# The dev tooling. `development` is a private repository, so it is staged
# into the build context by scripts/build_ingestion_job.sh from the copy the
# CI already checked out -- no credential is needed here, and none ends up
# in a layer of what is a public image. kli is public and cloned directly.
COPY .build/development ${KALISIO_TOOLING_DIR}/development
RUN git clone --depth 1 https://github.com/kalisio/kli.git ${KALISIO_TOOLING_DIR}/kli \
    && cd ${KALISIO_TOOLING_DIR}/kli \
    && yarn install --frozen-lockfile \
    && yarn cache clean \
    && chown -R mambauser:mambauser ${KALISIO_TOOLING_DIR} \
    # yarn ran as root and left XDG_CACHE_HOME behind owned by root, which
    # micromamba could then no longer write to.
    && rm -rf ${XDG_CACHE_HOME}

# The same environment `development/install.sh` writes into a new recruit's
# .bashrc, minus the two tokens: the image is public, so KALISIO_GITHUB_TOKEN
# and GITLAB_IRSN_TOKEN are supplied at run time
# (development/workspaces/services/knowledge/knowledge.enc.env locally, the
# namespace secret in the cluster). The urls are derived from them here so an
# interactive shell in this container behaves like a developer's.
RUN printf '%s\n' \
    '# Kalisio environment, as development/install.sh sets it up.' \
    'export DEVELOPMENT_DIR=${DEVELOPMENT_DIR:-/workspace}' \
    'export KALISIO_DEVELOPMENT_DIR=$DEVELOPMENT_DIR/kalisio' \
    'export KALISIO_DEVELOPMENT_JOBS_DIR=$DEVELOPMENT_DIR/kalisio' \
    'export IRSN_DEVELOPMENT_DIR=$DEVELOPMENT_DIR/irsn' \
    'export AIRBUS_DEVELOPMENT_DIR=$DEVELOPMENT_DIR/airbus' \
    'export DEVELOPMENT_BIN_DIR=/opt/kalisio/development/scripts' \
    'export PATH=$PATH:$DEVELOPMENT_BIN_DIR' \
    'export kli="node /opt/kalisio/kli/index.js"' \
    'if [ -n "${KALISIO_GITHUB_TOKEN:-}" ]; then' \
    '  export KALISIO_GITHUB_URL="https://oauth2:$KALISIO_GITHUB_TOKEN@github.com"' \
    'else' \
    '  export KALISIO_GITHUB_URL="ssh://git@github.com"' \
    'fi' \
    'if [ -n "${GITLAB_IRSN_TOKEN:-}" ]; then' \
    '  export GITLAB_IRSN_URL="https://oauth2:$GITLAB_IRSN_TOKEN@gitlab.asnr.fr"' \
    'else' \
    '  export GITLAB_IRSN_URL="ssh://git@gitlab.asnr.fr:30000"' \
    'fi' \
    > /etc/profile.d/kalisio.sh

# apt-get/npm run as root with HOME=/app, which can implicitly create /app
# owned by root; COPY --chown only chowns what it copies, not a pre-existing
# parent dir, so force ownership before copying the app code into it.
RUN mkdir -p ${HOME} && chown mambauser:mambauser ${HOME} \
    && ln -sf /etc/profile.d/kalisio.sh ${HOME}/.bashrc \
    # /opt belongs to root; the model is fetched as mambauser below.
    && mkdir -p /opt/models && chown -R mambauser:mambauser /opt/models

# Named explicitly rather than `COPY .`: the build context also holds the
# staged tooling above, which has no business being copied a second time.
COPY --chown=mambauser:mambauser ingestion ${HOME}/ingestion
COPY --chown=mambauser:mambauser environment.runtime.yml ${HOME}/environment.runtime.yml
WORKDIR ${HOME}
USER mambauser

# The repositories are cloned into a volume that belongs to another uid, and
# git refuses to read those ("detected dubious ownership"): every
# `git ls-files` would come back empty and the job would index nothing.
RUN git config --global --add safe.directory '*'

RUN micromamba install -y -n base -f environment.runtime.yml \
    && micromamba clean --all --yes

# Must match the EMBEDDING_MODEL the job is configured with, and the one the
# api queries with; another value still works, it is downloaded at first use.
ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RUN micromamba run -n base python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('${EMBEDDING_MODEL}')"

CMD ["micromamba", "run", "-n", "base", "python", "-m", "ingestion.bin"]
