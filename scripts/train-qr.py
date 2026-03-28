import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import VariationalELBO
from sklearn.preprocessing import MinMaxScaler

from model import MTGPQR_H, save_mtgpqr

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
args = parser.parse_args()

X = pd.read_csv(args.X)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float()
scaler = MinMaxScaler().fit(X)
X_scale = torch.tensor(scaler.scale_).float()
X_min = torch.tensor(scaler.min_).float()

X_scaled = scaler.transform(X)
inducing_points = torch.tensor(np.unique(X_scaled, axis=0)).float()
if args.target == "H":
    model_class = MTGPQR_H
else:
    raise NotImplementedError
model = model_class(inducing_points, X_scale, X_min)

# train
model.train()
mll = VariationalELBO(model.likelihood, model.gp, num_data=y.numel())
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

N_ITER = 10_000
for i in range(N_ITER):
    output = model.gp(torch.tensor(X_scaled).float())
    loss = -mll(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Iter {i + 1}/{N_ITER} - Loss: {loss.item():.3f}")

save_mtgpqr(model, args.out)
