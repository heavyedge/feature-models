# syntax=docker/dockerfile:1.19
FROM ghcr.io/astral-sh/uv:latest AS uv

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

FROM python:slim AS clear-notebooks

WORKDIR /src
COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv --no-cache pip install --system nbstripout
COPY notebooks ./notebooks
RUN nbstripout notebooks/*.ipynb


FROM python:slim AS dev

WORKDIR /src
COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git make openssl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /dataset/_data ./_data

COPY --from=clear-notebooks /src/notebooks ./notebooks
COPY --exclude=.git --exclude=notebooks . .

ARG IMAGE_CREATED
ARG IMAGE_VERSION
ARG IMAGE_REVISION
ARG IMAGE_TITLE
ARG IMAGE_DESCRIPTION
RUN mkdir -p /etc/heavyedge \
    && echo "${IMAGE_VERSION}" > /etc/heavyedge/image-version \
    && echo "${IMAGE_REVISION}" > /etc/heavyedge/image-revision
LABEL org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.authors="Jisoo Song <jeesoo9595@snu.ac.kr>" \
      org.opencontainers.image.documentation="https://heavyedge.github.io/feature-models/" \
      org.opencontainers.image.source="https://github.com/jisoosong/heavyedge/feature-models" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="HeavyEdge Feature Models (dev)" \
      org.opencontainers.image.description="Image for developing heavyedge/feature-models. Includes source in '/src' directory. Does not include trained models."
