import argparse
import importlib
import pathlib
import sys

import numpy as np
import torch

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

torch.manual_seed(42)


def predict(X, X_scaler, y_scaler, mean, model, chunks_size=4096):
    X_raw = torch.tensor(X, dtype=torch.float32, device=device)

    quantiles = []
    with torch.no_grad():
        for i in range(0, X_raw.shape[0], chunks_size):
            X_pred = X_raw[i : i + chunks_size]
            X_scaled = X_scaler(X_pred)
            scaled_res_quantiles = model.mean_quantiles_delta(X_scaled)
            pred_res = y_scaler.inverse_transform(scaled_res_quantiles)
            pred_mean = mean(X_pred).reshape(-1, 1)
            pred_quantiles = pred_res + pred_mean
            quantiles.append(pred_quantiles.cpu().numpy())
    return np.concatenate(quantiles, axis=0)


parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature npy file.")
parser.add_argument("model", type=pathlib.Path, help="Model pt file.")
parser.add_argument("--target", required=True)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npz file."
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == "H":
    loader = load_module.load_H_models
elif args.target == "phi":
    loader = load_module.load_phi_models
else:
    raise ValueError(f"Unknown target: {args.target}")

X = np.load(args.X)
quantiles, X_scaler, y_scaler, mean, _, model = loader(path=args.model, device=device)

X_flattened = X.reshape(-1, X.shape[-1])
with torch.no_grad():
    pred = predict(X_flattened, X_scaler, y_scaler, mean, model)
pred = pred.reshape(X.shape[:-1] + (pred.shape[-1],))
np.savez(args.out, quantile_levels=quantiles.cpu().numpy(), quantiles=pred)
