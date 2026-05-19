import argparse
import logging
import pathlib

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import model as model_module
from model import save_model, train_model

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", help="Model class prefix.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X = pd.read_csv(args.X)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float()
scaler = MinMaxScaler().fit(X)
X_scale = torch.tensor(scaler.scale_).float()
X_min = torch.tensor(scaler.min_).float()

X_scaled = torch.tensor(scaler.transform(X)).float()
model_cls_name = f"{args.model}_{args.target}"
model_class = getattr(model_module, model_cls_name)

model = train_model(
    X_scaled.to(device),
    y.to(device),
    model_class(X_scaled, X_scale, X_min).to(device),
    num_epochs=10_000,
    learning_rate=0.001,
    logger=lambda i, num_epochs, loss: logger.info(
        f"{args.out}: Epoch {i+1}/{num_epochs}, Loss: {loss:.4f}"
    ),
)
save_model(model, args.out)
