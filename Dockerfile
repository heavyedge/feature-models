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
