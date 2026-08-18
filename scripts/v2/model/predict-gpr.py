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
        " If not passed, the default prior mean model is used."
    ),
)
parser.add_argument(
    "gpr_model",
    type=pathlib.Path,
    nargs="?",
    help=(
        "Path to the gpr model file." " If not passed, the default GPR model is used."
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

prior_mean_loader = getattr(load_module, "load_PriorMean")
prior_mean_model = prior_mean_loader(path=args.prior_mean_model, device=device)
prior_mean_model.eval()

gpr_loader = getattr(load_module, "load_GPR")
X_scaler, y_scaler, likelihood, gpr_model = gpr_loader(
    path=args.gpr_model, device=device
)
X_scaler.eval()
y_scaler.eval()
gpr_model.eval()
likelihood.eval()

TARGET_COLUMNS = gpr_model.output_names
if prior_mean_model.output_names != TARGET_COLUMNS:
    parser.error(
        "prior-mean and GPR models have different output names: "
        f"{prior_mean_model.output_names} != {TARGET_COLUMNS}"
    )

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., i : i + args.chunk_size, :]

        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(X_chunk)
        scaled_res_posterior = gpr_model(X_scaled)

        chunk_size = X_chunk.shape[-2]
        expected_shape = X_chunk.shape[:-2] + (chunk_size, len(TARGET_COLUMNS))
        if prior_mean.shape != expected_shape:
            parser.error(
                "unexpected prior-mean output shape "
                f"{tuple(prior_mean.shape)}; expected {tuple(expected_shape)}"
            )
        if scaled_res_posterior.mean.shape != expected_shape:
            parser.error(
                "unexpected GPR posterior mean shape "
                f"{tuple(scaled_res_posterior.mean.shape)}; "
                f"expected {tuple(expected_shape)}"
            )
        if scaled_res_posterior.variance.shape != expected_shape:
            parser.error(
                "unexpected GPR posterior variance shape "
                f"{tuple(scaled_res_posterior.variance.shape)}; "
                f"expected {tuple(expected_shape)}"
            )

        residual_mean = y_scaler.inverse_transform(scaled_res_posterior.mean)
        residual_std = (
            scaled_res_posterior.variance.sqrt() * y_scaler.X_scale.abs().unsqueeze(-2)
        )

        posterior_mean = prior_mean + residual_mean
        chunk_result = torch.stack((posterior_mean, residual_std), dim=-1).cpu().numpy()
        result_shape = chunk_result.shape[:-1]  # (*B, N, T)
        batch_shape = chunk_result.shape[:-3]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(batch_shape + (1, 1)),
                result_shape,
            ).ravel()
        else:
            batch = np.full(np.prod(result_shape), "", dtype=object)

        data = {
            "index": np.broadcast_to(
                X_row_indices[..., i : i + chunk_size].reshape(
                    X_row_indices.shape[:-1] + (chunk_size, 1)
                ),
                result_shape,
            ).ravel(),
            "batch": batch,
            "target": np.broadcast_to(
                np.asarray(TARGET_COLUMNS).reshape(
                    (1,) * (posterior_mean.ndim - 1) + (-1,)
                ),
                result_shape,
            ).ravel(),
            "mean": chunk_result[..., 0].ravel(),
            "std": chunk_result[..., 1].ravel(),
        }

        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(columns=["index", "batch", "target", "mean", "std"]).to_csv(
        args.out, index=False
    )
