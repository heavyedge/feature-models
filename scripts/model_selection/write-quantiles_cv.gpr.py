import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from cv import quantiles_cv_gpr, split_data
from gpytorch.likelihoods import GaussianLikelihood

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
parser.add_argument("--model", required=True)
parser.add_argument("--target", required=True)
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
    help="Output csv file of CV of mean prediction.",
)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_csv(args.X).drop(columns="Slurry").values
y = pd.read_csv(args.y)[args.target].values

x_train_cv, y_train_cv, x_test_cv, y_test_cv, x_scales, x_mins = split_data(
    X, y, args.num_folds, device
)

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)
model_cls = getattr(model_module, args.model)

likelihood = GaussianLikelihood(batch_shape=torch.Size([args.num_folds])).to(device)
model = model_cls(
    x_train_cv.clone().detach(),
    y_train_cv.clone().detach(),
    likelihood,
    X_scale=x_scales,
    X_min=x_mins,
    batch_shape=torch.Size([args.num_folds]),
).to(device)

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

cv = quantiles_cv_gpr(
    x_train_cv,
    y_train_cv,
    x_test_cv,
    y_test_cv,
    quantiles,
    model,
    likelihood,
    n_epochs=args.n_epochs,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)

pd.DataFrame(cv).to_csv(args.out, index=False)
