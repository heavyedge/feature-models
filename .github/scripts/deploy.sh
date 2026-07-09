#!/bin/sh

set -eu

GITHUB_APP_TOKEN="$(sh .github/scripts/app-token.sh)"
export GITHUB_APP_TOKEN

sh .github/scripts/dispatch-cleanup.sh
