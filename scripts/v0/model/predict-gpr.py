import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

from . import load as load_module
from .batch import load_batched_features

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
parser.add_argument(
    "--batch-col",
    type=int,
    nargs="*",
    default=[],
    help=(
        "CSV column(s) defining batch dimensions. Each column becomes one "
        "batch dimension, and every combination of their values must have the "
        "same number of rows."
    ),
)
parser.add_argument("--target", required=True, nargs="+", choices=["H", "phi"])
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

try:
    X_values, X_row_indices = load_batched_features(
        args.X, args.index_col, args.batch_col
    )
except ValueError as exc:
    parser.error(str(exc))
X = torch.tensor(X_values, dtype=torch.float32, device=device)

if args.chunk_size <= 0:
    parser.error("--chunk-size must be positive")

prior_mean_loader = getattr(load_module, f"load_PriorMean_{args.target[0]}")
prior_mean_model = prior_mean_loader(path=args.prior_mean_model, device=device)
prior_mean_model.eval()

gpr_loader = getattr(load_module, f"load_GPR_{args.target[0]}")
X_scaler, y_scaler, likelihood, gpr_model = gpr_loader(
    path=args.gpr_model, device=device
)
X_scaler.eval()
y_scaler.eval()
gpr_model.eval()
likelihood.eval()

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., i : i + args.chunk_size, :]

        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(X_chunk)
        scaled_f_posterior = gpr_model(X_scaled)
        scaled_y_posterior = likelihood(scaled_f_posterior)

        latent_mean = prior_mean + y_scaler.inverse_transform(
            scaled_f_posterior.mean.unsqueeze(-1)
        ).squeeze(-1)
        latent_std = (
            scaled_f_posterior.variance.sqrt().unsqueeze(-1)
            * y_scaler.X_scale.abs().unsqueeze(-2)
        ).squeeze(-1)
        predictive_mean = prior_mean + y_scaler.inverse_transform(
            scaled_y_posterior.mean.unsqueeze(-1)
        ).squeeze(-1)
        predictive_std = (
            scaled_y_posterior.variance.sqrt().unsqueeze(-1)
            * y_scaler.X_scale.abs().unsqueeze(-2)
        ).squeeze(-1)
        chunk_result = (
            torch.stack(
                (latent_mean, latent_std, predictive_mean, predictive_std), dim=-1
            )
            .cpu()
            .numpy()
        )
        chunk_size = X_chunk.shape[-2]
        if latent_mean.shape[-1] == chunk_size:
            multitask = False
        elif latent_mean.ndim >= 2 and latent_mean.shape[-2] == chunk_size:
            multitask = True
        else:
            parser.error(f"unexpected model output shape {tuple(latent_mean.shape)}")
        num_tasks = latent_mean.shape[-1] if multitask else 1
        if len(args.target) != num_tasks:
            parser.error(
                f"--target requires {num_tasks} value(s) for this model; "
                f"got {len(args.target)}"
            )

        result_shape = chunk_result.shape[:-1]
        batch_shape = chunk_result.shape[: -3 if multitask else -2]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(
                    batch_shape + (1,) * (2 if multitask else 1)
                ),
                result_shape,
            ).ravel()
        else:
            batch = np.full(np.prod(result_shape), "", dtype=object)

        data = {
            "index": np.broadcast_to(
                X_row_indices[..., i : i + chunk_size].reshape(
                    X_row_indices.shape[:-1]
                    + (chunk_size,)
                    + ((1,) if multitask else ())
                ),
                result_shape,
            ).ravel(),
            "batch": batch,
            "target": np.broadcast_to(
                np.asarray(args.target).reshape(
                    (1,) * (latent_mean.ndim - (1 if multitask else 0))
                    + ((num_tasks,) if multitask else ())
                ),
                result_shape,
            ).ravel(),
            "latent_mean": chunk_result[..., 0].ravel(),
            "latent_std": chunk_result[..., 1].ravel(),
            "predictive_mean": chunk_result[..., 2].ravel(),
            "predictive_std": chunk_result[..., 3].ravel(),
        }

        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(
        columns=[
            "index",
            "batch",
            "target",
            "latent_mean",
            "latent_std",
            "predictive_mean",
            "predictive_std",
        ]
    ).to_csv(args.out, index=False)
