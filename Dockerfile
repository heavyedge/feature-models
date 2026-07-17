# syntax=docker/dockerfile:1.19
FROM ghcr.io/astral-sh/uv:latest AS uv


FROM python:slim AS downloader
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

COPY setup.sh .
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -s /run/secrets/hf_token ]; then \
        hf auth login --token "$(cat /run/secrets/hf_token)"; \
    fi \
    && ./setup.sh


FROM scratch as doc
WORKDIR /app

COPY doc ./doc
# If built document exists in the source directory, doc/build/html is copied.
# If not, create empty directory to aviod error when copying to the final image.
WORKDIR doc/build/html


FROM python:slim AS train

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git jq openssl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /app/_data ./_data
COPY . .

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
      org.opencontainers.image.title="HeavyEdge Feature Models (train)" \
      org.opencontainers.image.description="Training environment for heavyedge/feature-models. Includes source in '/app' directory. Does not include trained models."


FROM python:slim as infer
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app
COPY model ./model
COPY --from=doc /app/doc/build/html ./doc

WORKDIR /workdir

ARG IMAGE_CREATED
ARG IMAGE_VERSION
ARG IMAGE_REVISION
ARG IMAGE_TITLE
ARG IMAGE_DESCRIPTION
LABEL org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.authors="Jisoo Song <jeesoo9595@snu.ac.kr>" \
      org.opencontainers.image.documentation="https://heavyedge.github.io/feature-models/" \
      org.opencontainers.image.source="https://github.com/jisoosong/heavyedge/feature-models" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="HeavyEdge Feature Models (infer)" \
      org.opencontainers.image.description="Inference environment for heavyedge/feature-models. Includes trained models in '/app' directory. Does not include source."


FROM python:slim as dev
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app
COPY --from=train . .
COPY --from=infer . .

WORKDIR /workdir

ARG IMAGE_CREATED
ARG IMAGE_VERSION
ARG IMAGE_REVISION
ARG IMAGE_TITLE
ARG IMAGE_DESCRIPTION
LABEL org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.authors="Jisoo Song <jeesoo9595@snu.ac.kr>" \
      org.opencontainers.image.documentation="https://heavyedge.github.io/feature-models/" \
      org.opencontainers.image.source="https://github.com/jisoosong/heavyedge/feature-models" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="HeavyEdge Feature Models (dev)" \
      org.opencontainers.image.description="Development environment for heavyedge/feature-models. Includes source and trained models in '/app' directory."
