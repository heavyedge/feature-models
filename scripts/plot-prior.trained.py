import argparse
import pathlib

import gpytorch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import ScalarFormatter

import model as model_module
from model import load_model

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("checkpoint", type=pathlib.Path, help="Model weight file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model class name.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output image file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to run on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

X = pd.read_csv(args.X)
y = pd.read_csv(args.y)
data = pd.concat([X, y], axis=1)

model_cls_name = f"{args.model}_{args.target}"
model_cls = getattr(model_module, model_cls_name, None)
model, _, scaler = load_model(model_cls, args.checkpoint, device=device)
model.eval()

# Plot

groups = list(data.groupby(["Cos_theta"]))
Rgts = data["Gap_to_thickness_ratio"]
Rgt_pred = np.linspace(Rgts.min(), Rgts.max(), 100)
Cas = data["Capillary_number"]

fig, axes = plt.subplots(1, len(groups), sharex=True, sharey="row")
if len(groups) == 1:
    axes = [axes]

ca_unique = np.sort(Cas.unique())
n_colors = len(ca_unique)
cmap = plt.get_cmap("viridis", n_colors)
norm = mcolors.BoundaryNorm(
    np.concatenate(
        [
            [ca_unique[0] * 0.9],
            (ca_unique[:-1] + ca_unique[1:]) / 2,
            [ca_unique[-1] * 1.1],
        ]
    ),
    ncolors=n_colors,
)

for ax, ((cos_theta,), df) in zip(axes, groups):

    for Ca, sub_df in df.groupby("Capillary_number", observed=True, as_index=False):
        color = cmap(norm(Ca))
        ax.scatter(sub_df["Gap_to_thickness_ratio"], sub_df[args.target], color=color)

        X_pred = np.stack(
            [
                Rgt_pred,
                np.full_like(Rgt_pred, Ca),
                np.full_like(Rgt_pred, cos_theta),
            ],
            axis=-1,
        )
        X_pred_scaled = torch.tensor(scaler.transform(X_pred)).float().to(device)
        with torch.no_grad(), gpytorch.settings.prior_mode(state=True):
            prior_mean = model.center_mean(X_pred_scaled)
        q_central = prior_mean.cpu().numpy()
        ax.plot(
            Rgt_pred,
            q_central,
            color=color,
        )

    ax.set_title(f"Cos θ={cos_theta:.2f}")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=axes, orientation="horizontal", location="top", pad=0.2, aspect=30
)
cbar.set_label("Ca")
quartile_vals = np.quantile(Cas.values, [0, 0.25, 0.5, 0.75, 1.0])
nearest_cas = [ca_unique[np.argmin(np.abs(ca_unique - q))] for q in quartile_vals]
cbar.set_ticks([round(ca, 3) for ca in nearest_cas])
cbar.ax.xaxis.set_major_formatter(ScalarFormatter())

fig.supxlabel("Rgt")
fig.supylabel(args.target)

plt.savefig(args.out)
plt.close()
