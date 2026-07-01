import argparse
import importlib
import pathlib
import sys

import numpy as np
import torch


def predict(model, X_np, device, chunk_size=4096):
    pred_chunks = []
    for j in range(0, len(X_np), chunk_size):
        X_chunk = torch.tensor(
            X_np[j : j + chunk_size], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred_chunks.append(
                model.mean_quantiles_delta(X_chunk).detach().cpu().numpy()
            )
    return np.concatenate(pred_chunks, axis=0)


parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature npy file.")
parser.add_argument("model", type=pathlib.Path, help="Model pt file.")
parser.add_argument("--target", required=True)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npz file."
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

if args.target == "H":
    loader = load_module.load_H_quantiles
elif args.target == "phi":
    loader = load_module.load_phi_quantiles
else:
    raise ValueError(f"Unknown target: {args.target}")

X = np.load(args.X)
quantiles, model, _, scaler = loader(path=args.model, device=device)

X_scaled = scaler.transform(X.reshape(-1, X.shape[-1]))
with torch.no_grad():
    pred = predict(model, X_scaled, device)  # (N, Q)
pred = pred.reshape(X.shape[:-1] + (pred.shape[-1],))
np.savez(args.out, quantile_levels=quantiles.cpu().numpy(), quantiles=pred)
