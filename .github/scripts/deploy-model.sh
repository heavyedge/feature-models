#!/bin/sh
set -e
if [ ! -r /etc/heavyedge/image-revision ]; then
  echo "Missing image revision file: /etc/heavyedge/image-revision" >&2
  exit 1
fi
if [ "$(cat /etc/heavyedge/image-revision)" != "${GIT_SHA}" ]; then
  echo "Image revision mismatch: expected ${GIT_SHA}" >&2
  echo "Actual image revision:" >&2
  cat /etc/heavyedge/image-revision >&2
  exit 1
fi
pip install -r requirements.txt
HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make models
if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  python upload.py
fi
