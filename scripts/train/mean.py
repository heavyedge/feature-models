import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from save import save_gpr

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument("--prior-mean", type=str, help="Prior mean class name.")
parser.add_argument("--num-epochs", type=int, help="Number of training epochs.")
parser.add_argument(
    "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X = torch.tensor(pd.read_csv(args.X).drop(columns="Slurry").values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)[..., None]

X_scaler = model_module.MinMaxScaler().to(device)
y_scaler = model_module.StandardScaler().to(device)

X_scaler.train()
X_scaled = X_scaler(X)

if args.prior_mean is not None:
    mean_class = getattr(model_module, args.prior_mean)
else:
    mean_class = ZeroMean
mean = mean_class().to(device)

model_class = getattr(model_module, args.model)
likelihood = GaussianLikelihood().to(device)
with torch.no_grad():
    y_scaler.train()
    y_scaled = y_scaler(y - mean(X)).squeeze(-1)
model = model_class(X_scaled, y_scaled, likelihood).to(device)

model.train()
likelihood.train()

mll = ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(
    list(X_scaler.parameters())
    + list(y_scaler.parameters())
    + list(mean.parameters())
    + list(model.parameters()),
    lr=args.learning_rate,
)

for i in range(args.num_epochs):
    X_scaler.train()
    y_scaler.train()
    mean.train()
    model.train()
    likelihood.train()
    optimizer.zero_grad()

    X_scaled = X_scaler(X)
    res = y_scaler(y - mean(X)).squeeze(-1)
    model.set_train_data(
        inputs=X_scaled.detach(),
        targets=res.detach(),
        strict=False,
    )
    output = model(X_scaled)
    loss = -mll(output, res)
    loss.sum().backward()
    optimizer.step()

    with torch.no_grad():
        X_scaler.train()
        y_scaler.train()
        X_scaled = X_scaler(X)
        res = y_scaler(y - mean(X)).squeeze(-1)
        model.set_train_data(
            inputs=X_scaled,
            targets=res,
            strict=False,
        )

    logger.info(f"{args.out}: Epoch {i+1}/{args.num_epochs}, Loss: {loss.item():.4f}")

save_gpr(
    X,
    y,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    args.out,
)
