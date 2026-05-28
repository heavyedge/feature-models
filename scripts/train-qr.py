import argparse
import logging
import os
import pathlib

import gpytorch
import pandas as pd
import torch
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP
from sklearn.preprocessing import MinMaxScaler

import model as model_module
from model import save_model

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

QUANTILES = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
CENTER_QUANTILE_INDEX = 2
NUM_LOWER_QUANTILES = 2
NUM_LATENTS = 3
NUM_LOWER_LATENTS = 1

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", help="Model class prefix.")
parser.add_argument("--num-epochs", type=int, help="Number of training epochs.")
parser.add_argument(
    "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

NUM_EPOCHS = int(
    os.getenv(
        "HEAVYEDGE_N_EPOCHS",
        args.num_epochs if args.num_epochs is not None else 10_000,
    )
)

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X = pd.read_csv(args.X).drop(columns="Slurry")
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float()
scaler = MinMaxScaler()
X_scaled = torch.tensor(scaler.fit_transform(X.to_numpy())).float()
X_scale = torch.tensor(scaler.scale_).float()
X_min = torch.tensor(scaler.min_).float()

model_cls_name = f"{args.model}_{args.target}"
model_class = getattr(model_module, model_cls_name)
inducing_points = X_scaled.clone()
model = model_class(
    inducing_points=inducing_points,
    num_quantiles=len(QUANTILES),
    num_lower_quantiles=NUM_LOWER_QUANTILES,
    num_latents=NUM_LATENTS,
    num_lower_latents=NUM_LOWER_LATENTS,
    X_scale=X_scale,
    X_min=X_min,
).to(device)

if issubclass(model_class, CenterGapQuantileGP):
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        QUANTILES, CENTER_QUANTILE_INDEX
    ).to(device)
elif issubclass(model_class, DirectQuantileGP):
    likelihood = MultitaskQuantileGPLikelihood(QUANTILES).to(device)
else:
    raise ValueError(f"Unknown model class: {model_class}")

# Train
train_x = X_scaled.to(device)
train_y = y.to(device)

model.train()
likelihood.train()

parameters = list(model.parameters()) + list(likelihood.parameters())
optimizer = torch.optim.Adam(
    parameters,
    lr=args.learning_rate,
)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

for i in range(NUM_EPOCHS):
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    logger.info(f"{args.out}: Epoch {i+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

# Save
save_model(
    train_x,
    train_y,
    model,
    likelihood,
    scaler,
    inducing_points,
    QUANTILES,
    NUM_LOWER_QUANTILES,
    NUM_LATENTS,
    NUM_LOWER_LATENTS,
    args.out,
)
