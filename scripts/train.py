import argparse
import logging
import pathlib

import gpqr
import pandas as pd
import torch
from gpytorch.mlls import VariationalELBO
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
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

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("cv", type=pathlib.Path, help="Cross-validation csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model class name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

cv_df = pd.read_csv(args.cv)
fold_cols = [col for col in cv_df.columns if col.startswith("test_loss_fold")]
best_epochs_per_fold = cv_df[fold_cols].idxmin() + 1
NUM_EPOCHS = int(best_epochs_per_fold.median())

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

model_cls_name = f"{args.model}"
model_cls = getattr(gpqr, model_cls_name, None)
if model_cls is None:
    raise ValueError(f"Model class {model_cls_name} not found.")

mean_cls_name = f"PriorMean_{args.target}"
mean_cls = getattr(gpqr, mean_cls_name, None)
if mean_cls is None:
    raise ValueError(f"Mean class {mean_cls_name} not found.")

X = torch.tensor(pd.read_csv(args.X).values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

scaler = StandardScaler().fit(X.detach().cpu())
X_scale = torch.tensor(scaler.scale_).float().to(device)
X_mean = torch.tensor(scaler.mean_).float().to(device)

inducing_points = X.clone().detach()
if args.model == "CgLmcMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_lower_quantiles=NUM_LOWER_QUANTILES,
        num_latents=NUM_LATENTS,
        num_lower_latents=NUM_LOWER_LATENTS,
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
    ).to(device)
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        QUANTILES,
        CENTER_QUANTILE_INDEX,
        torch.zeros((len(QUANTILES),)),
        learn_scales=True,
    ).to(device)
elif args.model == "CgIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_lower_quantiles=NUM_LOWER_QUANTILES,
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
    ).to(device)
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        QUANTILES,
        CENTER_QUANTILE_INDEX,
        torch.zeros((len(QUANTILES),)),
        learn_scales=True,
    ).to(device)
elif args.model == "DirectLmcMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_latents=NUM_LATENTS,
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
    ).to(device)
    likelihood = MultitaskQuantileGPLikelihood(
        QUANTILES,
        torch.zeros((len(QUANTILES),)),
        learn_scales=True,
    ).to(device)
elif args.model == "DirectIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
    ).to(device)
    likelihood = MultitaskQuantileGPLikelihood(
        QUANTILES,
        torch.zeros((len(QUANTILES),)),
        learn_scales=True,
    ).to(device)

mll = VariationalELBO(likelihood, model, num_data=len(y))
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(likelihood.parameters()),
    lr=0.001,
)

for i in range(NUM_EPOCHS):
    model.train()
    likelihood.train()
    output = model(X)

    train_loss = -mll(output, y)
    train_loss.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    logger.info(
        f"{args.out}: Epoch {i+1}/{NUM_EPOCHS}, Train Loss: {train_loss.item():.4f}"
    )

torch.save(
    {
        "quantiles": QUANTILES,
        "inducing_points": inducing_points.cpu(),
        "center_quantile_index": CENTER_QUANTILE_INDEX,
        "num_lower_quantiles": NUM_LOWER_QUANTILES,
        "num_latents": NUM_LATENTS,
        "num_lower_latents": NUM_LOWER_LATENTS,
        "X_scale": X_scale.cpu(),
        "X_mean": X_mean.cpu(),
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
    },
    args.out,
)
