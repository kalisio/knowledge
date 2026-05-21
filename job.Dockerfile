FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . ${HOME}
WORKDIR ${HOME}

RUN pip install --no-cache-dir .

CMD ["python", "-m", "ingestion_job"]
