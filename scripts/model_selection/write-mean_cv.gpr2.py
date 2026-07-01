import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from cv import mean_cv_gpr2, split_data2
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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
parser.add_argument("--target", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--prior-mean", type=str, help="Prior mean class name.")
parser.add_argument(
    "--num-folds",
    type=int,
    required=True,
    help="Number of folds for cross-validation.",
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
    help="Output csv file of CV of mean prediction.",
)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor(pd.read_csv(args.X).drop(columns="Slurry").values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)[..., None]

x_train, y_train, x_test, y_test = split_data2(
    X.cpu().numpy(), y.cpu().numpy(), args.num_folds, device
)
x_scaler = model_module.MinMaxScaler(batch_shape=torch.Size([args.num_folds])).to(
    device
)
y_scaler = model_module.StandardScaler(batch_shape=torch.Size([args.num_folds])).to(
    device
)

x_scaler.train()
x_scaled = x_scaler(x_train)

if args.prior_mean is not None:
    mean_class = getattr(model_module, args.prior_mean)
else:
    mean_class = ZeroMean
mean = mean_class(batch_shape=torch.Size([args.num_folds])).to(device)

model_class = getattr(model_module, args.model)
likelihood = GaussianLikelihood(batch_shape=torch.Size([args.num_folds])).to(device)
model = model_class(
    x_scaled, y_train.squeeze(-1), likelihood, batch_shape=torch.Size([args.num_folds])
).to(device)

cv = mean_cv_gpr2(
    x_train,
    y_train,
    x_test,
    y_test,
    x_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    n_epochs=args.n_epochs,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)

pd.DataFrame(cv).to_csv(args.out, index=False)
