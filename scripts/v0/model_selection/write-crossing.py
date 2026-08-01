import argparse
import logging
import pathlib

import model.gpqr_other as model_module  # Needs PYTHONPATH=scripts/v0
import model.prior as prior_module  # Needs PYTHONPATH=scripts/v0
import model.scale as scaler_module  # Needs PYTHONPATH=scripts/v0
import pandas as pd
import torch
from crossing import quantile_crossing
from gpytorch_qr.likelihoods import DirectQuantileLikelihood

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
parser.add_argument(
    "X_test",
    type=pathlib.Path,
    nargs="+",
    help="Predictor csv files for testing.",
)
parser.add_argument("--target", required=True)
parser.add_argument("--model", required=True)
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantiles for the model.",
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
    help="Output csv file of quantile crossing.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor(pd.read_csv(args.X).values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

X_tests = []
for path in args.X_test:
    Xtest_df = pd.read_csv(path, index_col=[0, 1, 2])
    X_tests.append(torch.tensor(Xtest_df.values).float().to(device))

dim = X.shape[-1]
batch_shape = X.shape[:-2]

X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)

X_scaler.train()
X_scaled = X_scaler(X)

mean_class = getattr(prior_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))

quantiles = torch.tensor(args.quantiles, dtype=torch.float32).to(device)

model_class = getattr(model_module, args.model)
likelihood = DirectQuantileLikelihood(
    quantiles.unsqueeze(0),
    torch.zeros((*batch_shape, len(quantiles))),
    learn_scales=True,
).to(device)
model = model_class(
    inducing_points=X_scaled.clone().detach(),
    num_quantiles=len(quantiles),
    num_latents=args.num_latents,
    batch_shape=batch_shape,
).to(device)

crs, mcs, mxs = quantile_crossing(
    X,
    y,
    X_tests,
    X_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    n_epochs=args.n_epochs,
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
