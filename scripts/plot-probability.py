import argparse
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import load_mtgpqr

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("X_scaler", type=pathlib.Path, help="Scaler model pkl file for X.")
parser.add_argument("model", type=pathlib.Path, help="Trained model pth file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output plot file.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_csv(args.X)
X_scaler = joblib.load(args.X_scaler)

Rgt_Ca_mesh = np.stack(
    np.meshgrid(
        np.linspace(1.5, 2.5, 100),
        np.linspace(0.08, 0.18, 100),
        indexing="ij",
    ),
    axis=-1,
)

if args.target == "H":
    from model import PriorMean_H

    D = X.shape[1]
    mean_module = PriorMean_H(torch.ones(D).float(), torch.zeros(D).float())
else:
    raise NotImplementedError

model = load_mtgpqr(args.model, mean_module, device=device).eval()
taus = model.gp.taus.detach().cpu().numpy()


def quantile_interpolation(q_values, q_levels, threshold):
    idx = np.array([np.searchsorted(row, threshold) for row in q_values])
    idx_clamped = np.clip(idx, 1, len(q_levels) - 1)
    rows = np.arange(len(q_values))
    x0 = q_values[rows, idx_clamped - 1]
    x1 = q_values[rows, idx_clamped]
    y0 = q_levels[idx_clamped - 1]
    y1 = q_levels[idx_clamped]
    probs = y0 + (threshold - x0) * (y1 - y0) / (x1 - x0)
    probs = np.where(idx == 0, 0.0, probs)
    probs = np.where(idx == len(q_levels), 1.0, probs)
    return probs


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
    X_pred = X_scaler.transform(
        pd.DataFrame(
            np.concatenate([Rgt_Ca_mesh, st_arr], axis=-1).reshape(-1, 3),
            columns=X.columns,
        )
    )
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
