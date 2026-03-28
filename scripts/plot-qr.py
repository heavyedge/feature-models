import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import MTGPQR_H, quantile_interpolation

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("model", type=pathlib.Path, help="Trained model pth file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output plot file.")
args = parser.parse_args()

X = pd.read_csv(args.X)

mesh = np.stack(
    np.meshgrid(
        np.linspace(1.5, 2.5, 100),
        np.linspace(0.08, 0.18, 100),
        indexing="ij",
    ),
    axis=-1,
)

surface_tensions, X_preds = [], []
for st, _ in reversed(list(X.groupby("Surface_tension"))):
    surface_tensions.append(st)
    X_preds.append(np.concatenate([mesh, np.full(mesh.shape[:-1] + (1,), st)], axis=-1))

checkpoint = torch.load(args.model)
if args.target == "H":
    model_class = MTGPQR_H
    threshold = 1.2
else:
    raise NotImplementedError
model = model_class(checkpoint["inducing_points"])
model.load_state_dict(checkpoint["state_dict"])

fig, axes = plt.subplots(
    1,
    len(X_preds),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    figsize=(4, 3),
)
for ax, X_pred, st in zip(axes, X_preds, surface_tensions):
    X_pred_tensor = torch.tensor(X_pred.reshape(-1, 3)).float()
    with torch.no_grad():
        pred = model(X_pred_tensor).cpu().numpy()
    prob = quantile_interpolation(pred, model.taus.cpu().numpy(), threshold)

    contour = ax.contour(
        mesh[:, :, 0],
        mesh[:, :, 1],
        prob.reshape(mesh.shape[:-1]),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        levels=[0.05, 0.5, 0.95],
    )
    ax.set_title(f"{st:.2f}")

plt.savefig(args.out)
plt.close()
