import argparse
import logging
import pathlib

import gpqr
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import VariationalELBO
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
K = 5

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model class prefix.")
parser.add_argument("--n-epochs", type=int, help="Number of training epochs.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

model_cls_name = f"{args.model}_{args.target}"
model_cls = getattr(gpqr, model_cls_name, None)
if model_cls is None:
    raise ValueError(f"Model class {model_cls_name} not found.")

X = torch.tensor(pd.read_csv(args.X).values).float()
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float()

kf = KFold(n_splits=K, shuffle=True, random_state=42)
x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
x_scales, x_means = [], []
for train_idx, test_idx in kf.split(X):
    x_train_list.append(X[train_idx])
    y_train_list.append(y[train_idx])
    x_test_list.append(X[test_idx])
    y_test_list.append(y[test_idx])

    scaler = StandardScaler().fit(X[train_idx].detach().cpu())
    x_scales.append(torch.tensor(scaler.scale_).float())
    x_means.append(torch.tensor(scaler.mean_).float())

x_train_cv = torch.stack(x_train_list).to(device)
y_train_cv = torch.stack(y_train_list).to(device)
x_test_cv = torch.stack(x_test_list).to(device)
y_test_cv = torch.stack(y_test_list).to(device)
x_scales = torch.stack(x_scales).to(device)
x_means = torch.stack(x_means).to(device)

inducing_points = x_train_cv.clone().detach()
if args.model == "CgLmcMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_lower_quantiles=NUM_LOWER_QUANTILES,
        num_latents=NUM_LATENTS,
        num_lower_latents=NUM_LOWER_LATENTS,
        X_scale=x_scales,
        X_mean=x_means,
        batch_shape=(K,),
    ).to(device)
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        QUANTILES.unsqueeze(0),
        CENTER_QUANTILE_INDEX,
        torch.zeros((K, len(QUANTILES))),
        learn_scales=True,
    ).to(device)
elif args.model == "CgIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_lower_quantiles=NUM_LOWER_QUANTILES,
        X_scale=x_scales,
        X_mean=x_means,
        batch_shape=(K,),
    ).to(device)
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        QUANTILES.unsqueeze(0),
        CENTER_QUANTILE_INDEX,
        torch.zeros((K, len(QUANTILES))),
        learn_scales=True,
    ).to(device)
elif args.model == "DirectLmcMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_latents=NUM_LATENTS,
        X_scale=x_scales,
        X_mean=x_means,
        batch_shape=(K,),
    ).to(device)
    likelihood = MultitaskQuantileGPLikelihood(
        QUANTILES.unsqueeze(0),
        torch.zeros((K, len(QUANTILES))),
        learn_scales=True,
    ).to(device)
elif args.model == "DirectIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        X_scale=x_scales,
        X_mean=x_means,
        batch_shape=(K,),
    ).to(device)
    likelihood = MultitaskQuantileGPLikelihood(
        QUANTILES.unsqueeze(0),
        torch.zeros((K, len(QUANTILES))),
        learn_scales=True,
    ).to(device)

mll = VariationalELBO(likelihood, model, num_data=y_train_cv.shape[1])
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(likelihood.parameters()),
    lr=0.001,
)

test_losses = []
for i in range(args.n_epochs):
    model.train()
    likelihood.train()
    output = model(x_train_cv)

    train_loss = -mll(output, y_train_cv)
    train_loss.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        output = model.mean_quantiles_mc(x_test_cv)  # (K, N, Q)
        pinball_losses = []
        for j, q in enumerate(QUANTILES):
            test_loss = mean_pinball_loss(
                y_test_cv.cpu().numpy(), output[:, :, j].cpu().numpy(), alpha=q.item()
            )
            pinball_losses.append(test_loss)
        test_losses.append(np.mean(pinball_losses))

    logger.info(
        f"{args.out}: Epoch {i+1}/{args.n_epochs}, Test Loss: {test_losses[-1]:.4f}"
    )

df = pd.DataFrame({"epoch": np.arange(1, args.n_epochs + 1), "test_loss": test_losses})
df.to_csv(args.out, index=False)
