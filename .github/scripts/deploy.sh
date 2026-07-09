#!/bin/sh

set -eu

python .github/k8s/send-email.py --status "Started"

GITHUB_APP_TOKEN="$(sh .github/scripts/app-token.sh)"
export GITHUB_APP_TOKEN

sh .github/scripts/dispatch-cleanup.sh

python .github/k8s/send-email.py --status "Completed"
