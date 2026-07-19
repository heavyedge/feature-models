import argparse
import importlib
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from cv import cv_gpqr, split_data
from gpytorch_qr.likelihoods import CenterGapQuantileLikelihood

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)

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
    help="Prior mean model weight file.",
)
parser.add_argument("--target", required=True)
parser.add_argument("--model", required=True)
parser.add_argument(
    "--num-folds",
    type=int,
    required=True,
    help="Number of folds for cross-validation.",
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
    help="Output npy file of CV of quantile predictions.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor(pd.read_csv(args.X).values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

dim = X.shape[-1]
batch_shape = torch.Size([args.num_folds])

x_train, y_train, x_test, y_test = split_data(
    X.cpu().numpy(), y.cpu().numpy(), args.num_folds, device
)
X_scaler = model_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = model_module.StandardScaler(1, batch_shape=batch_shape).to(device)

X_scaler.train()
X_scaled = X_scaler(X)

mean_class = getattr(model_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
mean.eval()

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model_class = getattr(model_module, args.model)
likelihood = CenterGapQuantileLikelihood(
    quantiles.unsqueeze(0),
    args.num_lower_quantiles,
    torch.zeros((*batch_shape, len(quantiles))),
    learn_scales=True,
).to(device)
model = model_class(
    inducing_points=X_scaled.clone().detach(),
    num_quantiles=len(quantiles),
    num_lower_quantiles=args.num_lower_quantiles,
    num_latents=args.num_latents,
    batch_shape=batch_shape,
).to(device)

cv = cv_gpqr(
    x_train,
    y_train,
    x_test,
    y_test,
    X_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    quantiles,
    n_epochs=args.n_epochs,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)

np.save(args.out, cv)
