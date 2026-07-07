#!/bin/sh
set -eu

status="${1:-}"
if [ -z "${status}" ]; then
  echo "Usage: $0 <started|succeeded|failed>" >&2
  exit 2
fi

if [ -z "${SMTP_NOTIFY_TO:-}" ]; then
  echo "SMTP_NOTIFY_TO is empty; skipping ${status} notification."
  exit 0
fi

python3 .github/scripts/notify-deploy.py "${status}"
