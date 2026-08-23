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

parser = argparse.ArgumentParser(description="Predict all v1 prior means.")
parser.add_argument("X", type=pathlib.Path, help="Input feature CSV.")
parser.add_argument("model", type=pathlib.Path, nargs="?")
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

model = load_module.load_PriorMean(path=args.model, device=device)
model.eval()
targets = tuple(model.output_names)

wrote_output = False
with torch.no_grad():
    for start in range(0, X.shape[-2], args.chunk_size):
        X_chunk = X[..., start : start + args.chunk_size, :]
        values = model(X_chunk).cpu().numpy()  # (*K, 3, N)
        chunk_size = X_chunk.shape[-2]
        expected_shape = tuple(X_chunk.shape[:-2]) + (len(targets), chunk_size)
        if values.shape != expected_shape:
            parser.error(
                f"unexpected model output shape {values.shape}; "
                f"expected {expected_shape}"
            )

        result_shape = values.shape
        batch_shape = result_shape[:-2]
        if batch_shape:
            batch = np.broadcast_to(
                np.arange(np.prod(batch_shape)).reshape(batch_shape + (1, 1)),
                result_shape,
            ).ravel()
        else:
            batch = np.full(values.size, "", dtype=object)
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
            "value": values.ravel(),
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
    pd.DataFrame(columns=["index", "batch", "target", "value"]).to_csv(
        args.out, index=False
    )
