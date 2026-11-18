FROM python:3.11-slim-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

# uv installs the locked dependencies (uv.lock), as in the CI.
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/tmp/.cache
# The environment lives next to the code and is on PATH, so `python` is the
# project's interpreter without `uv run`.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

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
ENV PATH="/app/.venv/bin:${PATH}:/opt/kalisio/development/scripts"

# The charts run the pod as uid 57439 (runAsUser, fsGroup), the uid of the
# former micromamba base image: the user keeps it so the deployments and the
# workspace volume need no change.
RUN groupadd --gid 57439 knowledge \
    && useradd --uid 57439 --gid 57439 --home-dir /app --no-create-home knowledge

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
    && chown -R knowledge:knowledge ${KALISIO_TOOLING_DIR} \
    # yarn ran as root and left XDG_CACHE_HOME behind owned by root, which
    # the job user could then no longer write to.
    && rm -rf ${XDG_CACHE_HOME}

# kli is cloned from master, so a change there lands in this image without a
# commit moving in this repository -- which is how 1.6.0, moving the command
# line out of index.js and behind a main-module test, stopped k-clone dead.
# There is nothing to pin to: the v1.6.0 tag and the published package both
# predate the pnpm support the workspaces now need. So assert the contract
# instead. k-clone runs `node $KALISIO_DEVELOPMENT_DIR/kli/index.js`, and
# _install_tooling makes that path a symlink, which defeated the main-module
# test of 1.6.0 (and defeats the current one under --preserve-symlinks-main)
# -- so the check has to go through a link too, as the job runs it. An entry
# point that prints nothing here is a job that clones nothing and reports
# success, and that is worth failing a build over.
RUN ln -s ${KALISIO_TOOLING_DIR}/kli /tmp/kli-entrypoint-check \
    && node /tmp/kli-entrypoint-check/index.js --version | grep -q . \
    && rm /tmp/kli-entrypoint-check

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
RUN mkdir -p ${HOME} && chown knowledge:knowledge ${HOME} \
    && ln -sf /etc/profile.d/kalisio.sh ${HOME}/.bashrc \
    # /opt belongs to root; the model is fetched as the job user below.
    && mkdir -p /opt/models && chown -R knowledge:knowledge /opt/models

# Named explicitly rather than `COPY .`: the build context also holds the
# staged tooling above, which has no business being copied a second time.
WORKDIR ${HOME}
USER knowledge

# The repositories are cloned into a volume that belongs to another uid, and
# git refuses to read those ("detected dubious ownership"): every
# `git ls-files` would come back empty and the job would index nothing.
RUN git config --global --add safe.directory '*'

# Dependencies first, in their own layer: a code change does not reinstall
# torch. --no-dev leaves the test and research groups out.
COPY --chown=knowledge:knowledge pyproject.toml uv.lock .python-version ${HOME}/
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY --chown=knowledge:knowledge ingestion ${HOME}/ingestion

# Must match the EMBEDDING_MODEL the job is configured with, and the one the
# api queries with; another value still works, it is downloaded at first use.
ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('${EMBEDDING_MODEL}')"

CMD ["python", "-m", "ingestion.bin"]
