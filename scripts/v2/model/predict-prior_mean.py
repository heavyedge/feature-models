import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

from models.v2.feature_models.batch import load_batched_features

from . import load as load_module

TARGET_COLUMNS = ("H", "phi_1", "phi_3")

parser = argparse.ArgumentParser(description="Predict prior mean of shape features.")
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
    "model",
    type=pathlib.Path,
    nargs="?",
    help=("Path to the model file."),
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

loader = load_module.load_PriorMean
model = loader(path=args.model, device=device)
model.eval()

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        pred_mean = model(X[..., i : i + args.chunk_size, :]).detach().cpu().numpy()
        chunk_size = min(args.chunk_size, X.shape[-2] - i)
        expected_shape = X.shape[:-2] + (chunk_size, len(TARGET_COLUMNS))
        if pred_mean.shape != expected_shape:
            parser.error(
                "unexpected model output shape "
                f"{pred_mean.shape}; expected {expected_shape}"
            )

        batch_shape = pred_mean.shape[:-2]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(batch_shape + (1,)),
                X_row_indices[..., i : i + chunk_size].shape,
            ).ravel()
        else:
            batch = np.full(chunk_size, "", dtype=object)

        data = {
            "index": X_row_indices[..., i : i + chunk_size].ravel(),
            "batch": batch,
            "H": pred_mean[..., 0].ravel(),
            "phi_1": pred_mean[..., 1].ravel(),
            "phi_3": pred_mean[..., 2].ravel(),
        }

        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(columns=["index", "batch", *TARGET_COLUMNS]).to_csv(
        args.out, index=False
    )
