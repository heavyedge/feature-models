import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import torch

from . import load as load_module
from .batch import load_batched_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Predict posterior distribution of shape features quantiles using GPQR."
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
    "gpqr_model",
    type=pathlib.Path,
    nargs="?",
    help=(
        "Path to the gpqr model file."
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
parser.add_argument(
    "--num-samples",
    default=20,
    type=int,
    help="Number of samples to draw from the posterior distribution.",
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

gpqr_loader = getattr(load_module, "load_GPQR")
quantiles, X_scaler, y_scaler, likelihood, gpqr_model = gpqr_loader(
    path=args.gpqr_model, device=device
)
X_scaler.eval()
y_scaler.eval()
gpqr_model.eval()
likelihood.eval()

# This implementation requires that all quantiles are the same for all outputs.
assert len(set([tuple(q.detach().cpu().numpy()) for q in quantiles])) == 1
Q = len(quantiles[0])
T = len(quantiles)

nsamples = torch.Size([args.num_samples])
quantile_levels = quantiles[0].detach().cpu().numpy()

TARGET_COLUMNS = gpqr_model.output_names
if prior_mean_model.output_names != TARGET_COLUMNS:
    parser.error(
        "prior-mean and GPQR models have different output names: "
        f"{prior_mean_model.output_names} != {TARGET_COLUMNS}"
    )

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., i : i + args.chunk_size, :]

        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(X_chunk)
        scaled_res_posterior = gpqr_model.joint_quantile_posterior(X_scaled)
        scaled_res_posterior_samples = scaled_res_posterior.rsample(nsamples)
        # shape: (S, *B, N, Q*T) -> (S, *B, N, Q, T)
        scaled_res_posterior_samples = scaled_res_posterior_samples.view(
            *scaled_res_posterior_samples.shape[:-1], Q, T
        )

        y_scale = y_scaler.X_scale.reshape(
            (1,) + tuple(y_scaler.X_scale.shape[:-1]) + (1, 1, y_scaler.dim)
        )
        y_mean = y_scaler.X_mean.reshape(
            (1,) + tuple(y_scaler.X_mean.shape[:-1]) + (1, 1, y_scaler.dim)
        )
        res = scaled_res_posterior_samples * y_scale + y_mean
        samples = prior_mean.unsqueeze(-2) + res  # (S, *B, N, Q, T)
        samples_np = samples.detach().cpu().numpy()

        chunk_size = X_chunk.shape[-2]
        expected_shape = (
            (args.num_samples,)
            + tuple(X_chunk.shape[:-2])
            + (chunk_size, Q, len(TARGET_COLUMNS))
        )
        if samples_np.shape != expected_shape:
            parser.error(
                f"unexpected model output shape {samples_np.shape}; "
                f"expected {expected_shape}"
            )

        ndim = samples_np.ndim
        row_indices = X_row_indices[..., i : i + chunk_size]
        batch_shape = samples_np.shape[1:-3]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape((1,) + batch_shape + (1,) * 3),
                samples_np.shape,
            ).ravel()
        else:
            batch = np.full(samples_np.size, "", dtype=object)

        data = {
            "index": np.broadcast_to(
                row_indices.reshape((1,) + row_indices.shape + (1, 1)),
                samples_np.shape,
            ).ravel(),
            "batch": batch,
            "target": np.broadcast_to(
                np.asarray(TARGET_COLUMNS).reshape((1,) * (ndim - 1) + (-1,)),
                samples_np.shape,
            ).ravel(),
            "quantile": np.broadcast_to(
                quantile_levels.reshape((1,) * (ndim - 2) + (-1, 1)),
                samples_np.shape,
            ).ravel(),
            "sample": np.broadcast_to(
                np.arange(samples_np.shape[0]).reshape((-1,) + (1,) * (ndim - 1)),
                samples_np.shape,
            ).ravel(),
            "value": samples_np.ravel(),
        }
        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        logger.info("Wrote chunk %s:%s to %s", i, i + chunk_size, args.out)
        wrote_output = True

if not wrote_output:
    pd.DataFrame(
        columns=["index", "batch", "target", "quantile", "sample", "value"]
    ).to_csv(args.out, index=False)
