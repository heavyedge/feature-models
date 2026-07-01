FROM python:slim AS downloader
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /dataset

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

RUN curl -LsSf https://hf.co/cli/install.sh | bash

RUN --mount=type=secret,id=hf_token,required=false \
    if [ -s /run/secrets/hf_token ]; then \
        hf auth login --token "$(cat /run/secrets/hf_token)"; \
    fi \
    && hf download jeesoo9595/heavyedge-features-v1 --repo-type dataset --revision v1.3.0 --local-dir _data


FROM python:slim AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace

COPY --from=downloader /dataset/_data ./_data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ARG HEAVYEDGE_N_EPOCHS
RUN env ${HEAVYEDGE_N_EPOCHS:+HEAVYEDGE_N_EPOCHS=${HEAVYEDGE_N_EPOCHS}} make all notebooks


FROM scratch AS output
WORKDIR /

COPY --from=builder /workspace/model /workspace/notebooks ./
