import os
import shutil

from huggingface_hub import HfApi

from version import __version__

api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))

MODEL_VERSION = f"v{__version__}"
MAJOR_VERSION = __version__.split(".")[0]
REPO = f"jeesoo9595/heavyedge-features-v{MAJOR_VERSION}"

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
