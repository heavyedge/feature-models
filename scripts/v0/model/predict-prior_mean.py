import argparse
import pathlib

import numpy as np
import pandas as pd
import torch

from . import load as load_module

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
    "model_file",
    type=pathlib.Path,
    nargs="?",
    help=(
        "Path to the model file."
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

loader = getattr(load_module, f"load_PriorMean_{args.target}")
model = loader(path=args.model_file, device=device)
model.eval()

ret = []
with torch.no_grad():
    for i in range(0, X.shape[0], args.chunk_size):
        pred_mean = model(X[i : i + args.chunk_size])
        ret.append(pred_mean.detach().cpu().numpy())
ret = np.concatenate(ret, axis=0)
pd.DataFrame({args.target: ret}).to_csv(args.out, index=False)
