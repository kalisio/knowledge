FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers

COPY --chown=mambauser:mambauser . ${HOME}
WORKDIR ${HOME}
USER mambauser

RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean --all --yes

EXPOSE 8187

CMD ["micromamba", "run", "-n", "base", "python", "-m", "api.bin"]

