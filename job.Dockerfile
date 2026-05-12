FROM python:3.11-bookworm
LABEL maintainer="<contact@kalisio.xyz>"

ENV HOME=/app
ENV PYTHONUNBUFFERED=1

COPY . ${HOME}
WORKDIR ${HOME}

RUN pip install --no-cache-dir .

CMD ["python", "-m", "knowledge.ingestion_job"]
