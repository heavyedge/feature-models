import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import torch
from gpytorch_qr.settings import quantile_gap_lower_bound

from . import gpr as gpr_module
from . import load as load_module
from .batch import load_batched_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Predict with the v1 batched GPQR.")
parser.add_argument("X", type=pathlib.Path, help="Input feature CSV.")
parser.add_argument("prior_mean_model", type=pathlib.Path, nargs="?")
parser.add_argument("gpr_model", type=pathlib.Path, nargs="?")
parser.add_argument("gpqr_model", type=pathlib.Path, nargs="?")
parser.add_argument("--index-col", type=int, nargs="*")
parser.add_argument("--batch-col", type=int, nargs="*", default=[])
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--chunk-size", type=int, default=4096)
parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
args = parser.parse_args()

if args.num_samples <= 0:
    parser.error("--num-samples must be positive")
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
gpr_X_scaler, gpr_y_scaler, _, gpr_model = load_module.load_GPR(args.gpr_model, device)
(
    quantiles,
    gap_lower_bound,
    X_scaler,
    y_scaler,
    likelihood,
    gpqr_model,
) = load_module.load_GPQR(args.gpqr_model, device)
for module in (
    prior_mean_model,
    gpr_X_scaler,
    gpr_y_scaler,
    gpr_model,
    X_scaler,
    y_scaler,
    likelihood,
    gpqr_model,
):
    module.eval()
targets = tuple(gpqr_model.output_names)
if (
    tuple(prior_mean_model.output_names) != targets
    or tuple(gpr_model.output_names) != targets
):
    parser.error("prior-mean, GPR, and GPQR output names differ")
quantile_levels = quantiles.detach().cpu().numpy()
Q = len(quantiles)


def expand_output_batch(values):
    return values.unsqueeze(-3).expand(
        *values.shape[:-2], len(targets), *values.shape[-2:]
    )


def broadcast_y_stat(stat, samples):
    stat = stat.squeeze(-1)
    prefix_dims = samples.ndim - stat.ndim - 2
    return stat.reshape((1,) * prefix_dims + tuple(stat.shape) + (1, 1))


wrote_output = False
with torch.no_grad(), quantile_gap_lower_bound(gap_lower_bound):
    for start in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., start : start + args.chunk_size, :]
        prior_mean = prior_mean_model(X_chunk)
        X_expanded = expand_output_batch(X_chunk)
        gpr_scaled_posterior = gpr_model(gpr_X_scaler(X_expanded))
        gpr_mean = gpr_module.posterior_mean(
            prior_mean, gpr_scaled_posterior, gpr_y_scaler
        )
        X_scaled = X_scaler(X_expanded)
        scaled_posterior = gpqr_model.joint_quantile_posterior(X_scaled)
        scaled_samples = scaled_posterior.rsample(torch.Size([args.num_samples]))
        # GPQR tensor contract: (S, *K, 3, N, Q).
        samples = (
            scaled_samples * broadcast_y_stat(y_scaler.X_scale, scaled_samples)
            + broadcast_y_stat(y_scaler.X_mean, scaled_samples)
            + gpr_mean.unsqueeze(0).unsqueeze(-1)
        )
        samples_np = samples.cpu().numpy()

        chunk_size = X_chunk.shape[-2]
        expected_shape = (
            (args.num_samples,)
            + tuple(X_chunk.shape[:-2])
            + (len(targets), chunk_size, Q)
        )
        if samples_np.shape != expected_shape:
            parser.error(
                f"unexpected model output shape {samples_np.shape}; "
                f"expected {expected_shape}"
            )

        ndim = samples_np.ndim
        batch_shape = samples_np.shape[1:-3]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape((1,) + batch_shape + (1, 1, 1)),
                samples_np.shape,
            ).ravel()
        else:
            batch = np.full(samples_np.size, "", dtype=object)
        data = {
            "index": np.broadcast_to(
                X_row_indices[..., start : start + chunk_size].reshape(
                    (1,) + X_row_indices.shape[:-1] + (1, chunk_size, 1)
                ),
                samples_np.shape,
            ).ravel(),
            "batch": batch,
            "target": np.broadcast_to(
                np.asarray(targets).reshape((1,) * (1 + len(batch_shape)) + (-1, 1, 1)),
                samples_np.shape,
            ).ravel(),
            "quantile": np.broadcast_to(
                quantile_levels.reshape((1,) * (ndim - 1) + (-1,)),
                samples_np.shape,
            ).ravel(),
            "sample": np.broadcast_to(
                np.arange(args.num_samples).reshape((-1,) + (1,) * (ndim - 1)),
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
        logger.info("Wrote chunk %s:%s to %s", start, start + chunk_size, args.out)
        wrote_output = True

if not wrote_output:
    pd.DataFrame(
        columns=["index", "batch", "target", "quantile", "sample", "value"]
    ).to_csv(args.out, index=False)
