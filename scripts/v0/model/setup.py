import shutil
import subprocess
import sys


def setup(model_module_path):

    requirements = str(model_module_path / "requirements.txt")
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", "--python", sys.executable, "-r", requirements]
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements]
        )
