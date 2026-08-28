FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/tmp/.cache

# The embedding model lives in the image, not in a cache the pod has to fill:
# a pod that starts fresh would otherwise pull 1.2 GB from HuggingFace on
# every start, unauthenticated and rate-limited. Baked in, the model loads in
# seconds and the service depends on nothing outside the cluster.
ENV HF_HOME=/opt/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

# /opt belongs to root; the model is fetched as mambauser, so the directory
# has to be theirs first.
USER root
RUN mkdir -p /opt/models && chown -R mambauser:mambauser /opt/models

COPY --chown=mambauser:mambauser . ${HOME}
WORKDIR ${HOME}
USER mambauser

RUN micromamba install -y -n base -f environment.runtime.yml \
    && micromamba clean --all --yes

# Must match the EMBEDDING_MODEL the service is configured with; another
# value still works, it is downloaded at first use as before.
ARG EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RUN micromamba run -n base python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('${EMBEDDING_MODEL}')"

EXPOSE 8187

CMD ["micromamba", "run", "-n", "base", "python", "-m", "api.bin"]

