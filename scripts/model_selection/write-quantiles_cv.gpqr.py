import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from cv import quantiles_cv_gpqr, split_data
from gpytorch_qr.likelihoods import MultitaskCenterGapQuantileGPLikelihood

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
    "--num-lower-latents",
    type=int,
    required=True,
    help="Number of lower latents for the model.",
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
    help="Output csv file of quantile crossing.",
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

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model = model_cls(
    inducing_points=x_train_cv.clone().detach(),
    num_quantiles=len(quantiles),
    num_lower_quantiles=args.num_lower_quantiles,
    num_latents=args.num_latents,
    num_lower_latents=args.num_lower_latents,
    X_scale=x_scales,
    X_min=x_mins,
    batch_shape=torch.Size([args.num_folds]),
).to(device)
likelihood = MultitaskCenterGapQuantileGPLikelihood(
    quantiles.unsqueeze(0),
    args.num_lower_quantiles,
    torch.zeros((args.num_folds, len(quantiles))),
    learn_scales=True,
).to(device)

ev = quantiles_cv_gpqr(
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

pd.DataFrame(ev).to_csv(args.out, index=False)
