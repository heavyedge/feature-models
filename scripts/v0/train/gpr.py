import argparse
import logging
import pathlib

import model.gpr as model_module  # Needs PYTHONPATH=scripts/v0
import model.prior as prior_module  # Needs PYTHONPATH=scripts/v0
import model.scale as scaler_module  # Needs PYTHONPATH=scripts/v0
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from save import save_gpr

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument(
    "prior_mean",
    type=pathlib.Path,
    help="Prior mean model weight file.",
)
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", help="Model class prefix.")
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

X = torch.tensor(pd.read_csv(args.X).values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

dim = X.shape[-1]
num_data = X.shape[-2]
batch_shape = X.shape[:-2]

X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)

X_scaler.train()
X_scaled = X_scaler(X)

mean_class = getattr(prior_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
mean.eval()

model_class = getattr(model_module, args.model)
likelihood = GaussianLikelihood(batch_shape=batch_shape).to(device)
with torch.no_grad():
    y_scaler.train()
    res = y_scaler((y - mean(X)).unsqueeze(-1)).squeeze(-1)
model = model_class(X_scaled, res, likelihood, batch_shape=batch_shape).to(device)

mll = ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(
    list(X_scaler.parameters())
    + list(y_scaler.parameters())
    + list(model.parameters()),
    lr=args.learning_rate,
)

X_scaler.train()
y_scaler.train()
likelihood.train()
model.train()
for i in range(args.num_epochs):
    optimizer.zero_grad()

    X_scaled = X_scaler(X)
    with torch.no_grad():
        train_mean = mean(X)
    res = y_scaler((y - train_mean).unsqueeze(-1)).squeeze(-1)
    model.set_train_data(
        inputs=X_scaled.detach(),
        targets=res.detach(),
        strict=False,
    )

    output = model(X_scaled)
    loss = -mll(output, res)
    loss.mean().backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        logger.info(
            f"{args.out}: Epoch {i+1}/{args.num_epochs}, Loss: {loss.mean().item():.4f}"
        )

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
