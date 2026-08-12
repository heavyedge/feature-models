set -eu

uv pip install --system huggingface_hub
python upload.py "${GITHUB_REF_NAME}"
