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
ENV PATH="/app/.venv/bin:${PATH}"

# The embedding model lives in the image, not in a cache the pod has to fill:
# a pod that starts fresh would otherwise pull 1.2 GB from HuggingFace on
# every start, unauthenticated and rate-limited. Baked in, the model loads in
# seconds and the service depends on nothing outside the cluster.
ENV HF_HOME=/opt/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

# The charts run the pod as uid 57439 (runAsUser), the uid of the former
# micromamba base image: the user keeps it so the deployments need no change.
RUN groupadd --gid 57439 knowledge \
    && useradd --uid 57439 --gid 57439 --home-dir ${HOME} --no-create-home knowledge \
    && mkdir -p ${HOME} /opt/models \
    && chown -R knowledge:knowledge ${HOME} /opt/models

WORKDIR ${HOME}
USER knowledge

# Dependencies first, in their own layer: a code change does not reinstall
# torch. --no-dev leaves the test group out.
COPY --chown=knowledge:knowledge pyproject.toml uv.lock .python-version ${HOME}/
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY --chown=knowledge:knowledge . ${HOME}

# Must match the EMBEDDING_MODEL the service is configured with; another
# value still works, it is downloaded at first use as before.
ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('${EMBEDDING_MODEL}')"

# Same for the translation models, so the first query in either language
# does not pay for a download. ~300 MB each against the embedding model's
# 1.2 GB. One per direction, comma-separated, exactly as
# QUERY_TRANSLATION_MODEL takes them: opus-mt-fr-en reaches the English pages
# from a French question, opus-mt-en-fr the French pages from an English one.
# A model the service is configured with but the image lacks still works, it
# is downloaded at first use.
ARG QUERY_TRANSLATION_MODEL=Helsinki-NLP/opus-mt-fr-en,Helsinki-NLP/opus-mt-en-fr
RUN python -c \
    "from transformers import MarianMTModel, MarianTokenizer; \
     names = [n.strip() for n in '${QUERY_TRANSLATION_MODEL}'.split(',') if n.strip()]; \
     [(MarianTokenizer.from_pretrained(n), MarianMTModel.from_pretrained(n)) for n in names]"

EXPOSE 8187

CMD ["python", "-m", "api.bin"]

