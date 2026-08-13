import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import v0.model.load as load_module  # Needs PYTHONPATH=scripts
from gpytorch.models import ExactGP
from gpytorch_qr.models import QuantileGP
from sklearn.metrics import mean_pinball_loss

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("model_file", type=pathlib.Path, help="Saved model pt file.")
parser.add_argument(
    "--index-col", type=int, nargs="*", help="Index columns for X and y."
)
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument(
    "--quantile-levels", type=float, nargs="*", help="Quantile levels to evaluate."
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
parser.add_argument("--device", choices=["cpu", "cuda"])
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X_df = pd.read_csv(args.X, index_col=args.index_col)
folds = sorted(X_df.index.unique())
X_arr = np.stack([X_df.loc[fold] for fold in folds], axis=0)
X = torch.tensor(X_arr).float().to(device)  # (*B, N, D)

y_df = pd.read_csv(args.y, index_col=args.index_col)[args.target]
y_arr = np.stack([y_df.loc[fold] for fold in sorted(y_df.index.unique())], axis=0)
y = torch.tensor(y_arr).float().to(device)  # (*B, N)

loader = getattr(load_module, f"load_{args.model}")
ret = loader(args.model_file, device=device)
model = ret[-1]

model.eval()
if isinstance(model, ExactGP):
    quantiles = model.quantiles(
        X, torch.tensor(args.quantile_levels).to(device)
    )  # (*B, N, Q)
    y_np = y.cpu().numpy()
    quantiles_np = quantiles.detach().cpu().numpy()
    loss_df = pd.DataFrame(
        [
            {
                "fold": fold,
                "quantile_level": q,
                "loss": mean_pinball_loss(
                    y_fold,
                    quantiles_fold[:, i],
                    alpha=float(q),
                ),
            }
            for fold, y_fold, quantiles_fold in zip(folds, y_np, quantiles_np)
            for i, q in enumerate(args.quantile_levels)
        ]
    )
elif isinstance(model, QuantileGP):
    ...
else:
    raise ValueError(f"Unknown model type: {type(model)}")

loss_df.to_csv(args.out, index=False)
