import argparse
import logging
import pathlib

import model.gpr as model_module  # Needs PYTHONPATH=scripts/v0
import model.prior as prior_module  # Needs PYTHONPATH=scripts/v0
import model.scale as scaler_module  # Needs PYTHONPATH=scripts/v0
import numpy as np
import pandas as pd
import torch
from cv import cv_gpr, split_data
from gpytorch.likelihoods import GaussianLikelihood

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
    "--n-epochs",
    type=int,
    required=True,
    help="Number of training epochs.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output csv file of CV of quantile prediction.",
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
X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)

X_scaler.train()
X_scaled = X_scaler(X)

mean_class = getattr(prior_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
mean.eval()

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model_class = getattr(model_module, args.model)
likelihood = GaussianLikelihood(batch_shape=batch_shape).to(device)
with torch.no_grad():
    y_scaler.train()
    res = y_scaler((y - mean(X)).unsqueeze(-1)).squeeze(-1)
model = model_class(X_scaled, res, likelihood, batch_shape=batch_shape).to(device)

cv = cv_gpr(
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

n_epochs, n_folds = cv.shape[1:]
cv_df = pd.DataFrame(
    {
        "epoch": np.repeat(np.arange(1, n_epochs + 1), n_folds),
        "fold": np.tile(np.arange(1, n_folds + 1), n_epochs),
        "test_mll_loss": cv[0].reshape(-1),
        "test_pinball_loss": cv[1].reshape(-1),
    }
)
cv_df.to_csv(args.out, index=False)
