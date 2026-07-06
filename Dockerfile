# syntax=docker/dockerfile:1.19
FROM python:slim AS downloader
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /dataset

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

RUN curl -LsSf https://hf.co/cli/install.sh | bash

COPY setup.sh .
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -s /run/secrets/hf_token ]; then \
        hf auth login --token "$(cat /run/secrets/hf_token)"; \
    fi \
    && ./setup.sh


FROM python:slim AS build-models
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /dataset/_data ./_data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ARG HEAVYEDGE_N_EPOCHS
RUN env ${HEAVYEDGE_N_EPOCHS:+HEAVYEDGE_N_EPOCHS=${HEAVYEDGE_N_EPOCHS}} make models


FROM python:slim AS build-notebooks
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /dataset/_data ./_data
COPY --from=build-models /workspace/model ./model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ARG HEAVYEDGE_N_EPOCHS
RUN env ${HEAVYEDGE_N_EPOCHS:+HEAVYEDGE_N_EPOCHS=${HEAVYEDGE_N_EPOCHS}} make notebooks


FROM scratch AS models
WORKDIR /

COPY --from=build-models /workspace/model ./


FROM scratch AS notebooks
WORKDIR /

COPY --from=build-notebooks /workspace/notebooks ./


FROM python:slim as dev-notebooks

WORKDIR /src
RUN pip install --no-cache-dir nbconvert
COPY notebooks ./notebooks
RUN jupyter nbconvert --clear-output --inplace notebooks/*.ipynb


FROM python:slim AS dev

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /dataset/_data ./_data

COPY --from=downloader /root/.local/bin/hf /root/.local/bin/hf
ENV PATH="/root/.local/bin:$PATH"

COPY --from=dev-notebooks /src/notebooks ./notebooks
COPY --exclude=notebooks . .

ARG IMAGE_CREATED
ARG IMAGE_VERSION
ARG IMAGE_REVISION
ARG IMAGE_TITLE
ARG IMAGE_DESCRIPTION
LABEL org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.authors="Jisoo Song <jeesoo9595@snu.ac.kr>" \
      org.opencontainers.image.source="https://github.com/jisoosong/heavyedge" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.title="${IMAGE_TITLE}" \
      org.opencontainers.image.description="${IMAGE_DESCRIPTION}"
