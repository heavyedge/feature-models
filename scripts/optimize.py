import argparse
import pathlib

import gpqr
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

QUANTILES = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model class prefix.")
parser.add_argument("--n-epochs", type=int, help="Number of training epochs.")
parser.add_argument("--n-trials", type=int, help="Number of optimization trials.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

model_cls_name = f"{args.model}_{args.target}"
model_cls = getattr(gpqr, model_cls_name, None)
if model_cls is None:
    raise ValueError(f"Model class {model_cls_name} not found.")

X = torch.tensor(pd.read_csv(args.X).values).float()
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float()

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
x_scales, x_means = [], []
for train_idx, test_idx in kf.split(X):
    x_train_list.append(X[train_idx])
    y_train_list.append(y[train_idx])
    x_test_list.append(X[test_idx])
    y_test_list.append(y[test_idx])

    scaler = StandardScaler().fit(X[train_idx].detach().cpu())
    x_scales.append(torch.tensor(scaler.scale_).float())
    x_means.append(torch.tensor(scaler.mean_).float())

x_train_cv = torch.stack(x_train_list).to(device)
y_train_cv = torch.stack(y_train_list).to(device)
x_test_cv = torch.stack(x_test_list).to(device)
y_test_cv = torch.stack(y_test_list).to(device)
x_scales = torch.stack(x_scales).to(device)
x_means = torch.stack(x_means).to(device)
