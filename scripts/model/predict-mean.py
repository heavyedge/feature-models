import argparse
import importlib
import pathlib
import sys

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))

parser = argparse.ArgumentParser(
    description="Predict prior mean from a trained model.",
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
    setup_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.setup")
    setup_module.setup(MODEL_MODULE_PATH)

    import numpy as np
    import torch

load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == "H":
    load_models = load_module.load_GPQR_H
elif args.target == "phi":
    load_models = load_module.load_GPQR_phi
models = load_models(device=device)
for module in models[1:]:
    module.eval()
_, _, _, mean, _, _ = models

X = np.load(args.X)
X_flattened = torch.tensor(
    X.reshape(-1, X.shape[-1]), dtype=torch.float32, device=device
)

ret = []
with torch.no_grad():
    for i in range(0, X_flattened.shape[0], args.chunk_size):
        X_pred = X_flattened[i : i + args.chunk_size]
        pred_mean = mean(X_pred)
        ret.append(pred_mean.cpu().numpy())
ret = np.concatenate(ret, axis=0)

np.save(args.out, ret.reshape(*X.shape[:-1]))
