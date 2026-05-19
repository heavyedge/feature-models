import argparse
import pathlib

import gpqr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import ScalarFormatter

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

model_cls_name = f"{args.model}"
model_cls = getattr(gpqr, model_cls_name, None)
if model_cls is None:
    raise ValueError(f"Model class {model_cls_name} not found.")

mean_cls_name = f"PriorMean_{args.target}"
mean_cls = getattr(gpqr, mean_cls_name, None)
if mean_cls is None:
    raise ValueError(f"Mean class {mean_cls_name} not found.")

X = pd.read_csv(args.X)
y = pd.read_csv(args.y)
data = pd.concat([X, y], axis=1)

checkpoint = torch.load(args.checkpoint, map_location=device)

inducing_points = checkpoint["inducing_points"].to(device)
QUANTILES = checkpoint["quantiles"]
NUM_LOWER_QUANTILES = checkpoint["num_lower_quantiles"]
NUM_LATENTS = checkpoint["num_latents"]
NUM_LOWER_LATENTS = checkpoint["num_lower_latents"]
X_scale = checkpoint["X_scale"].to(device)
X_mean = checkpoint["X_mean"].to(device)

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
elif args.model == "CgIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        num_lower_quantiles=NUM_LOWER_QUANTILES,
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
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
elif args.model == "DirectIndependentMtgpqr":
    model = model_cls(
        inducing_points=inducing_points,
        num_quantiles=len(QUANTILES),
        mean_cls=mean_cls,
        X_scale=X_scale,
        X_mean=X_mean,
    ).to(device)

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
        with torch.no_grad():
            quantiles = model.mean_quantiles_mc(torch.tensor(X_pred).float().to(device))
        q_low, q_high = quantiles.T[[0, -1], ...].cpu().numpy()
        ax.fill_between(
            Rgt_pred,
            q_low,
            q_high,
            facecolor=color,
            edgecolor="none",
            alpha=0.3,
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
