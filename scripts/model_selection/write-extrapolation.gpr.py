import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from cv import cross_validate_gpr, split_extrapolate_data
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
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantiles for the model.",
)
parser.add_argument(
    "--split-ratio",
    type=float,
    required=True,
    help="Ratio for splitting the data into training and testing sets.",
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

x_train_ev, y_train_ev, x_test_ev, y_test_ev, x_scales_ev, x_mins_ev = (
    split_extrapolate_data(X, y, args.split_ratio, device)
)

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)
model_cls = getattr(model_module, args.model)

likelihood = GaussianLikelihood(batch_shape=torch.Size([1])).to(device)
model = model_cls(
    x_train_ev.clone().detach(),
    y_train_ev.clone().detach(),
    likelihood,
    batch_shape=torch.Size([1]),
).to(device)

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

ev = cross_validate_gpr(
    x_train_ev,
    y_train_ev,
    x_test_ev,
    y_test_ev,
    quantiles,
    model,
    likelihood,
    n_epochs=args.n_epochs,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)

pd.DataFrame(ev).to_csv(args.out, index=False)
