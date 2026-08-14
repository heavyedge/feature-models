import argparse
import pathlib

import pandas as pd
import torch

from . import load as load_module

parser = argparse.ArgumentParser(
    description="Predict predictive posterior distribution of shape features using GPR."
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
parser.add_argument(
    "prior_mean_model",
    type=pathlib.Path,
    nargs="?",
    help=(
        "Path to the prior mean model file."
        "If not passed, default model will be searched using --target option."
    ),
)
parser.add_argument(
    "gpr_model",
    type=pathlib.Path,
    nargs="?",
    help=(
        "Path to the gpr model file."
        "If not passed, default model will be searched using --target option."
    ),
)
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns for X.")
parser.add_argument("--target", required=True, choices=["H", "phi"])
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_df = pd.read_csv(args.X, index_col=args.index_col if args.index_col else None)
X = torch.tensor(X_df.values, dtype=torch.float32, device=device)

prior_mean_loader = getattr(load_module, f"load_PriorMean_{args.target}")
prior_mean_model = prior_mean_loader(path=args.prior_mean_model, device=device)
prior_mean_model.eval()

gpr_loader = getattr(load_module, f"load_GPR_{args.target}")
X_scaler, y_scaler, likelihood, gpr_model = gpr_loader(
    path=args.gpr_model, device=device
)
X_scaler.eval()
y_scaler.eval()
gpr_model.eval()
likelihood.eval()

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[0], args.chunk_size):
        X_chunk = X[i : i + args.chunk_size]

        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(X_chunk)
        scaled_res_posterior = gpr_model(X_scaled)

        scaled_res_mean = scaled_res_posterior.mean.unsqueeze(-1)
        residual_mean = y_scaler.inverse_transform(scaled_res_mean).squeeze(-1)
        residual_std = (
            scaled_res_posterior.variance.sqrt().unsqueeze(-1)
            * y_scaler.X_scale.abs().unsqueeze(-2)
        ).squeeze(-1)

        posterior_mean = prior_mean + residual_mean
        chunk_result = torch.stack((posterior_mean, residual_std), dim=-1).cpu().numpy()
        pd.DataFrame(chunk_result, columns=["mean", "std"]).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(columns=["mean", "std"]).to_csv(args.out, index=False)
