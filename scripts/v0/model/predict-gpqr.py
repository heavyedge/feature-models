import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

from . import load as load_module

parser = argparse.ArgumentParser(
    description="Predict posterior distribution of shape features quantiles using GPR."
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
parser.add_argument("--target", required=True, choices=["H", "phi"])
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

X_raw_df = pd.read_csv(args.X)
X_df = pd.read_csv(args.X, index_col=args.index_col if args.index_col else None)

batch_col = args.batch_col
try:
    batch_keys = X_raw_df.iloc[:, batch_col]
except IndexError:
    parser.error(
        f"--batch-col contains a column outside the input range "
        f"[0, {X_raw_df.shape[1] - 1}]"
    )

# Each selected column is its own batch dimension. ``factorize`` keeps the
# order in which values first appear, including a single level for NaNs.
batch_codes = []
batch_shape = []
for column_index in range(batch_keys.shape[1]):
    codes, levels = pd.factorize(batch_keys.iloc[:, column_index], sort=False)
    if (codes == -1).any():
        codes = np.where(codes == -1, len(levels), codes)
        batch_shape.append(len(levels) + 1)
    else:
        batch_shape.append(len(levels))
    batch_codes.append(codes)

batch_shape = tuple(batch_shape)
batch_ids = np.zeros(len(X_df), dtype=int)
num_batches = 1
for codes, size in zip(reversed(batch_codes), reversed(batch_shape)):
    batch_ids += codes * num_batches
    num_batches *= size

batch_sizes = np.bincount(batch_ids, minlength=num_batches)
if batch_shape and ((batch_sizes == 0).any() or len(set(batch_sizes)) != 1):
    parser.error(
        "--batch-col must define a complete grid whose every batch has the "
        "same number of rows; "
        f"got batch sizes {batch_sizes.tolist()}"
    )

# Stable sorting keeps the original row order within each batch. Reshaping then
# yields (*B, N, D), where an empty *B is the ordinary (N, D) input shape.
order = np.argsort(batch_ids, kind="stable")
batch_size = int(batch_sizes[0])
X_values = X_df.iloc[order].values.reshape(*batch_shape, batch_size, X_df.shape[1])
X_row_indices = order.reshape(*batch_shape, batch_size)

X = torch.tensor(X_values, dtype=torch.float32, device=device)

prior_mean_loader = getattr(load_module, f"load_PriorMean_{args.target}")
prior_mean_model = prior_mean_loader(path=args.prior_mean_model, device=device)
prior_mean_model.eval()

gpqr_loader = getattr(load_module, f"load_GPQR_{args.target}")
quantiles, X_scaler, y_scaler, likelihood, gpqr_model = gpqr_loader(
    path=args.gpqr_model, device=device
)
X_scaler.eval()
y_scaler.eval()
gpqr_model.eval()
likelihood.eval()

nsamples = torch.Size([args.num_samples])
quantile_levels = quantiles.detach().cpu().numpy()

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., i : i + args.chunk_size, :]

        prior_mean = prior_mean_model(X_chunk)
        X_scaled = X_scaler(X_chunk)
        scaled_res_posterior = gpqr_model.joint_quantile_posterior(X_scaled)
        scaled_res_posterior_samples = scaled_res_posterior.rsample(nsamples)

        res = y_scaler.inverse_transform(scaled_res_posterior_samples)
        samples = prior_mean.unsqueeze(-1) + res  # (S, *B, N, Q)
        samples_np = samples.detach().cpu().numpy()

        ndim = samples_np.ndim
        row_indices = X_row_indices[..., i : i + samples_np.shape[-2]]
        data = {
            "index": np.broadcast_to(
                row_indices.reshape((1,) + row_indices.shape + (1,)),
                samples_np.shape,
            ).ravel(),
            "quantile": np.broadcast_to(
                quantile_levels.reshape((1,) * (ndim - 1) + (-1,)),
                samples_np.shape,
            ).ravel(),
            "sample": np.broadcast_to(
                np.arange(samples_np.shape[0]).reshape((-1,) + (1,) * (ndim - 1)),
                samples_np.shape,
            ).ravel(),
            args.target: samples_np.ravel(),
        }
        df = pd.DataFrame(data)
        df.to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(columns=["index", "quantile", "sample", args.target]).to_csv(
        args.out, index=False
    )
