import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import load_model, quantile_interpolation

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("model", type=pathlib.Path, help="Trained model pth file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output plot file.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_csv(args.X)

Rgt_Ca_mesh = np.stack(
    np.meshgrid(
        np.linspace(1.5, 2.5, 100),
        np.linspace(0.08, 0.18, 100),
        indexing="ij",
    ),
    axis=-1,
)

if args.target == "H":
    from model import MTGPQR_H as model_class
else:
    raise NotImplementedError

model = load_model(model_class, args.model, device=device).eval()
taus = model.taus.detach().cpu().numpy()


groups = list(reversed(list(X.groupby("Surface_tension"))))

fig, axes = plt.subplots(
    1,
    len(groups),
    sharex=True,
    sharey="row",
    constrained_layout=True,
    figsize=(4, 3),
)
for ax, (st, df) in zip(axes, groups):
    st_arr = np.full(Rgt_Ca_mesh.shape[:-1] + (1,), st)
    X_pred = np.concatenate([Rgt_Ca_mesh, st_arr], axis=-1).reshape(-1, 3)
    X_pred = torch.tensor(X_pred).float().to(device)
    with torch.no_grad():
        pred = model(X_pred).detach().cpu().numpy()
    H_prob = quantile_interpolation(pred, taus, threshold=1.2)
    prob = H_prob.reshape(Rgt_Ca_mesh.shape[:-1])

    contour = ax.contour(
        Rgt_Ca_mesh[:, :, 0],
        Rgt_Ca_mesh[:, :, 1],
        prob,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        levels=[0.05, 0.5, 0.95],
    )
    ax.clabel(contour, contour.levels)
    ax.set_title(st)

plt.savefig(args.out)
plt.close()
