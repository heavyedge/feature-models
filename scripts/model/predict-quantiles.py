import argparse
import importlib
import pathlib
import sys

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Predict quantiles from a trained model.",
)
parser.add_argument(
    "X",
    type=pathlib.Path,
    help=(
        "Input npy file, shape: (*B, N, D). "
        "The first three dimensions must be "
        "the Gap-to-thickness ratio, "
        "the Capillary number, and "
        "the cosine of the contact angle of the fluid on the substrate."
    ),
)
parser.add_argument("--target", required=True, choices=["H", "phi"])
parser.add_argument("--method", required=True, choices=["delta", "mc"])
parser.add_argument(
    "--num-samples",
    type=int,
    default=10,
    help="Number of MC samples when using the 'mc' method.",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=4096,
    help="Number of samples to process at once.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npy file."
)
args = parser.parse_args()

try:
    import numpy as np
    import torch
except ImportError:
    import shutil
    import subprocess

    requirements = str(MODEL_MODULE_PATH / "requirements.txt")
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", "--python", sys.executable, "-r", requirements]
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements]
        )

    import numpy as np
    import torch

sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == "H":
    load_models = load_module.load_H_models
elif args.target == "phi":
    load_models = load_module.load_phi_models
models = load_models(device=device)
for module in models[1:]:
    module.eval()
_, X_scaler, y_scaler, mean, likelihood, model = models

X = np.load(args.X)
X_flattened = torch.tensor(
    X.reshape(-1, X.shape[-1]), dtype=torch.float32, device=device
)

if args.method == "delta":
    quantiles = model.mean_quantiles_delta
elif args.method == "mc":

    def quantiles(x):
        return model.mean_quantiles_mc(x, num_samples=args.num_samples)

else:
    raise ValueError(f"Unknown method: {args.method}")

ret = []
with torch.no_grad():
    for i in range(0, X_flattened.shape[0], args.chunk_size):
        X_pred = X_flattened[i : i + args.chunk_size]
        X_scaled = X_scaler(X_pred)
        scaled_res_quantiles = quantiles(X_scaled)
        pred_res = y_scaler.inverse_transform(scaled_res_quantiles)
        pred_mean = mean(X_pred).reshape(-1, 1)
        pred_quantiles = pred_res + pred_mean
        ret.append(pred_quantiles.cpu().numpy())
ret = np.concatenate(ret, axis=0)

np.save(args.out, ret.reshape(*X.shape[:-1], -1))
