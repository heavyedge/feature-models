import argparse
import logging
import pathlib

import model.prior as model_module  # Needs PYTHONPATH=scripts/v0
import numpy as np
import pandas as pd
import torch

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
    "--index-col", type=int, nargs="*", help="Index columns for X and y."
)
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
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

X_df = pd.read_csv(args.X, index_col=args.index_col)
X_arr = np.stack([X_df.loc[fold] for fold in sorted(X_df.index.unique())], axis=0)
X = torch.tensor(X_arr).float().to(device)  # (*K, N, D)
print(X.shape)

y_df = pd.read_csv(args.y, index_col=args.index_col)[args.target]
y_arr = np.stack([y_df.loc[fold] for fold in sorted(y_df.index.unique())], axis=0)
y = torch.tensor(y_arr).float().to(device)  # (*K, N)
print(y.shape)

dim = X.shape[-1]
batch_shape = X.shape[:-2]

PriorMean = getattr(model_module, args.model)
model = PriorMean(batch_shape=batch_shape).to(device)

# Train
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.MSELoss()
for epoch in range(args.num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}] Loss: {loss.item():.6f}")


torch.save(
    model.state_dict(),
    args.out,
)
