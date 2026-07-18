import argparse
import importlib
import json
import os
import shutil
import sys

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    from setuptools._vendor.packaging.version import InvalidVersion, Version

parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
parser.add_argument("tag", help="Model version tag (e.g., v1.0.0)")
parser.add_argument(
    "--metadata-file",
    help="Write uploaded model metadata as JSON after a successful upload",
)
args = parser.parse_args()

version_text = args.tag.removeprefix("v")

try:
    version = Version(version_text)
except InvalidVersion:
    print(f"Invalid model version tag: {args.tag}", file=sys.stderr)
    sys.exit(1)

if version.post is not None:
    print(f"Skipping Hugging Face upload for post release tag: {args.tag}")
    sys.exit(1)
if version.dev is not None:
    print(f"Skipping Hugging Face upload for dev release tag: {args.tag}")
    sys.exit(1)

HfApi = importlib.import_module("huggingface_hub").HfApi
api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))

MODEL_VERSION = args.tag
MAJOR_VERSION = f"v{version.major}"
REPO = f"jeesoo9595/heavyedge-features-{MAJOR_VERSION}"

shutil.rmtree("model/__pycache__", ignore_errors=True)

api.create_repo(
    repo_id=REPO,
    repo_type="model",
    exist_ok=True,
)
api.upload_folder(
    folder_path="model",
    repo_id=REPO,
    repo_type="model",
    commit_message=f"Upload model version {MODEL_VERSION}",
)
api.create_tag(
    repo_id=REPO,
    tag=MODEL_VERSION,
)

if args.metadata_file:
    metadata = {
        "model_repo": f"{REPO}",
        "model_revision": MODEL_VERSION,
    }
    with open(args.metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, sort_keys=True)
        file.write("\n")
