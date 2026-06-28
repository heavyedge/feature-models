import argparse
import importlib
import logging
import os
import pathlib
import sys

import pandas as pd
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from save import save_model
from sklearn.preprocessing import MinMaxScaler

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.gpr")

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

model_cls_name = f"GPR_{args.target}"
model_class = getattr(model_module, model_cls_name)

likelihood = GaussianLikelihood().to(device)
model = model_class(X_scaled, y, likelihood, X_scale, X_min).to(device)

train_x = X_scaled.to(device)
train_y = y.to(device)

model.train()
likelihood.train()

mll = ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

for i in range(NUM_EPOCHS):
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    logger.info(f"{args.out}: Epoch {i+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

save_model(
    train_x,
    train_y,
    model,
    likelihood,
    scaler,
    None,
    None,
    None,
    None,
    None,
    args.out,
)
