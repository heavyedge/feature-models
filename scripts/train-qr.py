import argparse
import logging
import pathlib

import joblib
import pandas as pd
import torch

from model import PriorMean_H, save_mtgpqr, train_mtgpqr

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Predictor csv file.")
parser.add_argument("y", type=pathlib.Path, help="Loss csv file.")
parser.add_argument("X_scaler", type=pathlib.Path, help="Scaler model pkl file for X.")
parser.add_argument("--target", required=True, help="Field name.")
parser.add_argument(
    "--taus", type=float, nargs="+", required=True, help="Quantile levels."
)
parser.add_argument(
    "--num-epochs", type=int, required=True, help="Number of training epochs."
)
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X_scaler = joblib.load(args.X_scaler)

X = torch.tensor(X_scaler.transform(pd.read_csv(args.X))).float().to(device)
y = torch.tensor(pd.read_csv(args.y).values.reshape(-1)).float().to(device)
taus = torch.tensor(sorted(args.taus)).float().to(device)

if args.target == "H":
    mean_module = PriorMean_H(
        torch.tensor(X_scaler.scale_).float(),
        torch.tensor(X_scaler.min_).float(),
    )
else:
    raise NotImplementedError

torch.manual_seed(0)
model = train_mtgpqr(
    X,
    y,
    median_mean_module=mean_module,
    taus=taus,
    num_epochs=args.num_epochs,
    device=device,
    logger=lambda i, num_epochs, loss: logger.info(
        f"{args.out}: Epoch {i+1}/{num_epochs}, Loss: {loss:.4f}"
    ),
    num_half_lmc_latents=4,
)
save_mtgpqr(model, args.out)
