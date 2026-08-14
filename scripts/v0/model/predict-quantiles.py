import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))

parser = argparse.ArgumentParser(
    description="Predict quantiles from a trained model.",
)
parser.add_argument(
    "X",
    type=pathlib.Path,
    help=(
        "Input csv file, shape: (N, D). "
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
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == "H":
    load_mean = load_module.load_PriorMean_H
    load_models = load_module.load_GPQR_H
elif args.target == "phi":
    load_mean = load_module.load_PriorMean_phi
    load_models = load_module.load_GPQR_phi
mean = load_mean(device=device)
mean.eval()
models = load_models(device=device)
for module in models:
    try:
        module.eval()
    except AttributeError:
        pass
quantile_levels, X_scaler, y_scaler, likelihood, model = models

X = torch.tensor(
    pd.read_csv(args.X, index_col=[0, 1, 2]).values,
    dtype=torch.float32,
    device=device,
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
    for i in range(0, X.shape[0], args.chunk_size):
        X_pred = X[i : i + args.chunk_size]
        X_scaled = X_scaler(X_pred)
        scaled_res_quantiles = quantiles(X_scaled)
        pred_res = y_scaler.inverse_transform(scaled_res_quantiles)
        pred_mean = mean(X_pred).reshape(-1, 1)
        pred_quantiles = pred_res + pred_mean
        ret.append(pred_quantiles.cpu().numpy())
ret = np.concatenate(ret, axis=0)  # (N, n_quantiles)

quantile_levels = quantile_levels.detach().cpu().numpy()
pd.DataFrame(
    {
        f"{args.target} ({quantile_level:.0%})": ret[:, i]
        for i, quantile_level in enumerate(quantile_levels)
    }
).to_csv(args.out, index=False)
