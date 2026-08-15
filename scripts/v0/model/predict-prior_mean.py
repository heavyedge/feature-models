import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

from . import load as load_module
from .batch import load_batched_features

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
    help=(
        "Path to the model file."
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

loader = getattr(load_module, f"load_PriorMean_{args.target[0]}")
model = loader(path=args.model, device=device)
model.eval()

wrote_output = False
with torch.no_grad():
    for i in range(0, X.shape[-2], args.chunk_size):
        pred_mean = model(X[..., i : i + args.chunk_size, :]).detach().cpu().numpy()
        chunk_size = min(args.chunk_size, X.shape[-2] - i)
        if pred_mean.shape[-1] == chunk_size:
            multitask = False
        elif pred_mean.ndim >= 2 and pred_mean.shape[-2] == chunk_size:
            multitask = True
        else:
            parser.error(f"unexpected model output shape {pred_mean.shape}")
        num_tasks = pred_mean.shape[-1] if multitask else 1
        if len(args.target) != num_tasks:
            parser.error(
                f"--target requires {num_tasks} value(s) for this model; "
                f"got {len(args.target)}"
            )

        result_shape = pred_mean.shape
        batch_shape = pred_mean.shape[: -2 if multitask else -1]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(
                    batch_shape + (1,) * (2 if multitask else 1)
                ),
                result_shape,
            ).ravel()
        else:
            batch = np.full(pred_mean.size, "", dtype=object)

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
                    (1,) * (pred_mean.ndim - (1 if multitask else 0))
                    + ((num_tasks,) if multitask else ())
                ),
                result_shape,
            ).ravel(),
            "value": pred_mean.ravel(),
        }

        pd.DataFrame(data).to_csv(
            args.out,
            index=False,
            mode="a" if wrote_output else "w",
            header=not wrote_output,
        )
        wrote_output = True

if not wrote_output:
    pd.DataFrame(columns=["index", "batch", "target", "value"]).to_csv(
        args.out, index=False
    )
