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

parser = argparse.ArgumentParser(description="Predict with the v1 batched GPR.")
parser.add_argument("X", type=pathlib.Path, help="Input feature CSV.")
parser.add_argument("prior_mean_model", type=pathlib.Path, nargs="?")
parser.add_argument("gpr_model", type=pathlib.Path, nargs="?")
parser.add_argument("--index-col", type=int, nargs="*")
parser.add_argument("--batch-col", type=int, nargs="*", default=[])
parser.add_argument("--chunk-size", type=int, default=4096)
parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
args = parser.parse_args()

if args.chunk_size <= 0:
    parser.error("--chunk-size must be positive")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    X_values, X_row_indices = load_batched_features(
        args.X, args.index_col, args.batch_col
    )
except ValueError as exc:
    parser.error(str(exc))
X = torch.tensor(X_values, dtype=torch.float32, device=device)

prior_mean_model = load_module.load_PriorMean(args.prior_mean_model, device)
X_scaler, y_scaler, likelihood, gpr_model = load_module.load_GPR(args.gpr_model, device)
for module in (prior_mean_model, X_scaler, y_scaler, likelihood, gpr_model):
    module.eval()
targets = tuple(gpr_model.output_names)
if tuple(prior_mean_model.output_names) != targets:
    parser.error("prior-mean and GPR output names differ")


def expand_output_batch(values):
    return values.unsqueeze(-3).expand(
        *values.shape[:-2], len(targets), *values.shape[-2:]
    )


wrote_output = False
with torch.no_grad():
    for start in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., start : start + args.chunk_size, :]
        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(expand_output_batch(X_chunk))
        latent = gpr_model(X_scaled)
        predictive = likelihood(latent)

        latent_mean = prior_mean + y_scaler.inverse_transform(
            latent.mean.unsqueeze(-1)
        ).squeeze(-1)
        predictive_mean = prior_mean + y_scaler.inverse_transform(
            predictive.mean.unsqueeze(-1)
        ).squeeze(-1)
        y_scale = y_scaler.X_scale.abs().unsqueeze(-2).squeeze(-1)
        latent_std = latent.variance.sqrt() * y_scale
        predictive_std = predictive.variance.sqrt() * y_scale
        result = (
            torch.stack(
                (latent_mean, latent_std, predictive_mean, predictive_std), dim=-1
            )
            .cpu()
            .numpy()
        )

        chunk_size = X_chunk.shape[-2]
        result_shape = result.shape[:-1]  # (*K, 3, N)
        expected_shape = tuple(X_chunk.shape[:-2]) + (len(targets), chunk_size)
        if result_shape != expected_shape:
            parser.error(
                f"unexpected model output shape {result_shape}; "
                f"expected {expected_shape}"
            )
        batch_shape = result_shape[:-2]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(batch_shape + (1, 1)),
                result_shape,
            ).ravel()
        else:
            batch = np.full(np.prod(result_shape), "", dtype=object)
        data = {
            "index": np.broadcast_to(
                X_row_indices[..., start : start + chunk_size].reshape(
                    X_row_indices.shape[:-1] + (1, chunk_size)
                ),
                result_shape,
            ).ravel(),
            "batch": batch,
            "target": np.broadcast_to(
                np.asarray(targets).reshape((1,) * len(batch_shape) + (-1, 1)),
                result_shape,
            ).ravel(),
            "latent_mean": result[..., 0].ravel(),
            "latent_std": result[..., 1].ravel(),
            "predictive_mean": result[..., 2].ravel(),
            "predictive_std": result[..., 3].ravel(),
        }
        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        logger.info("Wrote chunk %s:%s to %s", start, start + chunk_size, args.out)
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
