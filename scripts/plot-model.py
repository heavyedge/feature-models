import argparse
import pathlib

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import load_mtgpqr

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("X_scaler", type=pathlib.Path, help="Scaler model pkl file for X.")
parser.add_argument("model", type=pathlib.Path, help="Trained model pth file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output plot file.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = pd.read_csv(args.X)
X_scaler = joblib.load(args.X_scaler)
y = pd.read_csv(args.y)
data = pd.concat([X, y], axis=1)

Rgts = X["Gap_to_thickness_ratio"]
Rgt_pred = np.linspace(Rgts.min(), Rgts.max(), 100)
Cas = X["Capillary_number"]

if args.target == "H":
    from model import PriorMean_H

    mean_module = PriorMean_H(X_scaler)
else:
    raise NotImplementedError

model = load_mtgpqr(args.model, mean_module, device=device).eval()

groups = list(reversed(list(data.groupby("Surface_tension"))))

fig, axes = plt.subplots(
    1,
    len(groups),
    sharex=True,
    sharey="row",
    constrained_layout=True,
    figsize=(4, 3),
)
norm = mcolors.LogNorm(vmin=Cas.min(), vmax=Cas.max())
cmap = plt.get_cmap("viridis")
for ax, (st, df) in zip(axes, groups):
    for Ca, sub_df in df.groupby("Capillary_number", observed=True, as_index=False):
        color = cmap(norm(Ca))

        ax.scatter(
            sub_df["Gap_to_thickness_ratio"], sub_df[args.target], color=color, s=1
        )

    ax.set_title(st)

    for Ca in Cas.unique():
        color = cmap(norm(Ca))
        X_pred = np.concatenate(
            [
                Rgt_pred.reshape(-1, 1),
                np.repeat([[Ca, st]], len(Rgt_pred), axis=0),
            ],
            axis=1,
        )
        X_pred = X_scaler.transform(pd.DataFrame(X_pred, columns=X.columns))
        X_pred = torch.tensor(X_pred).float().to(device)

        with torch.no_grad():
            quantiles = model(X_pred).cpu().detach().numpy()
        q_low, q_high = quantiles.T[[0, -1], ...]
        ax.fill_between(
            Rgt_pred,
            q_low,
            q_high,
            facecolor=color,
            edgecolor="none",
            alpha=0.3,
        )

plt.savefig(args.out)
plt.close()
