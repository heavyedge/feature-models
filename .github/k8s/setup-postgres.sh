#!/bin/sh

python3 -c 'import psycopg2' 2>/dev/null || pip install --no-cache-dir psycopg2-binary

python3 - "$OPTUNA_DB" <<'PY'
import sys
import time

import optuna
from sqlalchemy import create_engine

engine = create_engine(sys.argv[1])
for attempt in range(60):
    try:
        with engine.connect():
            break
    except Exception as error:
        if attempt == 59:
            raise RuntimeError("PostgreSQL sidecar did not become ready") from error
        time.sleep(1)

# Initialize Optuna's schema once before parallel model-training processes
# connect. This avoids concurrent CREATE TYPE/TABLE races on an empty database.
optuna.storages.RDBStorage(sys.argv[1])
PY

touch /var/run/heavyedge/optuna.ready