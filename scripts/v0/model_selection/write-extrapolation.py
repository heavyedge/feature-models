import argparse
import logging
import pathlib

import model.gpqr as model_module  # Needs PYTHONPATH=scripts/v0
import model.prior as prior_module  # Needs PYTHONPATH=scripts/v0
import model.scale as scaler_module  # Needs PYTHONPATH=scripts/v0
import numpy as np
import pandas as pd
import torch
from cv import cv_gpqr, split_extrapolate_data
from gpytorch.means import ZeroMean
from gpytorch_qr.likelihoods import CenterGapQuantilesLikelihood

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "X",
    type=pathlib.Path,
    help="Predictor csv file.",
)
parser.add_argument(
    "y",
    type=pathlib.Path,
    help="Response csv file.",
)
parser.add_argument(
    "prior_mean",
    type=pathlib.Path,
    nargs="?",
    help="Prior mean model weight file.",
)
parser.add_argument("--target", required=True)
parser.add_argument("--model", required=True)
parser.add_argument(
    "--split-ratio",
    type=float,
    required=True,
    help="Ratio for splitting the data into training and testing sets.",
)
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantiles for the model.",
)
parser.add_argument(
    "--num-lower-quantiles",
    type=int,
    required=True,
    help="Number of lower quantiles for the model.",
)
parser.add_argument(
    "--num-latents",
    type=int,
    required=True,
    help="Number of latents for the model.",
)
parser.add_argument(
    "--n-epochs",
    type=int,
    required=True,
    help="Number of training epochs.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output csv file of extrapolation CV.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor(pd.read_csv(args.X).values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

dim = X.shape[-1]
batch_shape = torch.Size([1])

x_train, y_train, x_test, y_test = split_extrapolate_data(
    X.cpu().numpy(), y.cpu().numpy(), args.split_ratio, device
)
x_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)

x_scaler.train()
y_scaler.train()
x_scaled = x_scaler(x_train)

if args.prior_mean is not None:
    mean_class = getattr(prior_module, "PriorMean_" + args.target)
    mean = mean_class(batch_shape=batch_shape).to(device)
    mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
else:
    mean = ZeroMean(batch_shape=batch_shape).to(device)
mean.eval()

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model_class = getattr(model_module, args.model)
likelihood = CenterGapQuantilesLikelihood(
    quantiles.unsqueeze(0),
    args.num_lower_quantiles,
    batch_shape=batch_shape,
).to(device)
model = model_class(
    inducing_points=x_scaled.clone().detach(),
    num_quantiles=len(quantiles),
    num_lower_quantiles=args.num_lower_quantiles,
    num_latents=args.num_latents,
    batch_shape=batch_shape,
).to(device)

ev = cv_gpqr(
    x_train,
    y_train,
    x_test,
    y_test,
    x_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    quantiles,
    n_epochs=args.n_epochs,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)
_, n_epochs, n_folds = ev.shape
ev_df = pd.DataFrame(
    {
        "epoch": np.repeat(np.arange(1, n_epochs + 1), n_folds),
        "fold": np.tile(np.arange(1, n_folds + 1), n_epochs),
        "test_mll_loss": ev[0].reshape(-1),
        "test_pinball_loss": ev[1].reshape(-1),
    }
)
ev_df.to_csv(args.out, index=False)
