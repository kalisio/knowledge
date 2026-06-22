FROM mambaorg/micromamba
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1

COPY --chown=mambauser:mambauser . ${HOME}
WORKDIR ${HOME}
USER mambauser

RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean --all --yes

EXPOSE 8000

CMD ["micromamba", "run", "-n", "base", "python", "-m", "api.main"]

