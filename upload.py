import os
import shutil

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))

REPO = "jeesoo9595/heavyedge-features-v1"
MODEL_VERSION = "v1.0.0"

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
