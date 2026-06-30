import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from crossing import quantile_crossing
from gpytorch_qr.likelihoods import MultitaskQuantileGPLikelihood
from sklearn.preprocessing import MinMaxScaler

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
parser.add_argument(
    "X_test",
    type=pathlib.Path,
    nargs="+",
    help="Predictor csv files for testing.",
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

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)
model_cls = getattr(model_module, args.model)

scaler = MinMaxScaler()
X = scaler.fit_transform(pd.read_csv(args.X).drop(columns="Slurry").values)
y = pd.read_csv(args.y)[args.target].values

X_scale = scaler.scale_
X_min = scaler.min_

X_tests = [scaler.transform(pd.read_csv(f).values) for f in args.X_test]

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)
X_scale = torch.tensor(X_scale, dtype=torch.float32).to(device)
X_min = torch.tensor(X_min, dtype=torch.float32).to(device)
X_tests = [torch.tensor(X_test, dtype=torch.float32).to(device) for X_test in X_tests]

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model = model_cls(
    inducing_points=X.clone().detach(),
    num_quantiles=len(quantiles),
    num_lower_quantiles=args.num_lower_quantiles,
    num_latents=args.num_latents,
    num_lower_latents=args.num_lower_latents,
    X_scale=X_scale,
    X_min=X_min,
).to(device)
likelihood = MultitaskQuantileGPLikelihood(
    q=quantiles,
    raw_scales=torch.zeros((len(quantiles),)),
    learn_scales=True,
).to(device)

crs, mcs, mxs = quantile_crossing(
    X,
    y,
    X_tests,
    model,
    likelihood,
    n_epochs=args.n_epochs,
    learning_rate=0.001,
    logger=lambda msg: logger.info(f"{args.out}: {msg}"),
)

data = dict()
for i, cr in enumerate(crs):
    data[f"crossing_rate_{i}"] = cr
for i, mc in enumerate(mcs):
    data[f"mean_crossing_{i}"] = mc
for i, mx in enumerate(mxs):
    data[f"max_crossing_{i}"] = mx

pd.DataFrame(data).to_csv(args.out, index=False)
