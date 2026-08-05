import argparse
import logging
import pathlib

import pandas as pd
import torch
import v0.model.prior as prior_module  # Needs PYTHONPATH=scripts
import v0.model.scale as scaler_module  # Needs PYTHONPATH=scripts
import v1.model.gpr as model_module  # Needs PYTHONPATH=scripts
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from v0.train.save import save_gpr  # Needs PYTHONPATH=scripts

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("Xtrain", type=pathlib.Path, help="Training feature csv file.")
parser.add_argument("ytrain", type=pathlib.Path, help="Training target csv file.")
parser.add_argument("Xval", type=pathlib.Path, help="Validation feature csv file.")
parser.add_argument("yval", type=pathlib.Path, help="Validation target csv file.")
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

Xtrain = torch.tensor(pd.read_csv(args.Xtrain).values).float().to(device)
ytrain = torch.tensor(pd.read_csv(args.ytrain)[args.target].values).float().to(device)
Xval = torch.tensor(pd.read_csv(args.Xval).values).float().to(device)
yval = torch.tensor(pd.read_csv(args.yval)[args.target].values).float().to(device)

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
batch_shape = Xtrain.shape[:-2]

X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)

X_scaler.train()
Xtrain_scaled = X_scaler(Xtrain)

mean_class = getattr(prior_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
mean.eval()

model_class = getattr(model_module, args.model)
likelihood = GaussianLikelihood(batch_shape=batch_shape).to(device)
with torch.no_grad():
    y_scaler.train()
    res = y_scaler((ytrain - mean(Xtrain)).unsqueeze(-1)).squeeze(-1)
model = model_class(Xtrain_scaled, res, likelihood, batch_shape=batch_shape).to(device)

mll = ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(
    list(X_scaler.parameters())
    + list(y_scaler.parameters())
    + list(model.parameters()),
    lr=args.learning_rate,
)

for epoch in range(args.num_epochs):
    X_scaler.train()
    y_scaler.train()
    likelihood.train()
    model.train()
    optimizer.zero_grad()

    Xtrain_scaled = X_scaler(Xtrain)
    with torch.no_grad():
        train_mean = mean(Xtrain)
    res = y_scaler((ytrain - train_mean).unsqueeze(-1)).squeeze(-1)
    model.set_train_data(
        inputs=Xtrain_scaled.detach(),
        targets=res.detach(),
        strict=False,
    )

    output = model(Xtrain_scaled)
    train_loss = -mll(output, res)
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        X_scaler.eval()
        y_scaler.eval()
        likelihood.eval()
        model.eval()
        Xval_scaled = X_scaler(Xval)
        val_mean = mean(Xval)
        val_res = y_scaler((yval - val_mean).unsqueeze(-1)).squeeze(-1)
        val_output = model(Xval_scaled)
        val_loss = -mll(val_output, val_res)

    if (epoch + 1) % 100 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss: {train_loss.item():.6f}, "
            f"Validation Loss: {val_loss:.6f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
        )

save_gpr(
    Xtrain,
    ytrain,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    args.out,
)
