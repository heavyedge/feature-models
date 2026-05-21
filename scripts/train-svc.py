import argparse
import pathlib
import pickle

import numpy as np
import optuna
import pandas as pd
from kernels import AnisotropicMatern52 as kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Observed X csv file.")
parser.add_argument("window", type=pathlib.Path, help="Quality window npy file.")
parser.add_argument(
    "--n-trials", type=int, default=100, help="Number of Optuna trials per slurry."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output pkl file.")
args = parser.parse_args()

X = pd.read_csv(args.X).drop(columns="Slurry").to_numpy()
W = np.load(args.window)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, x, y):
    n_features = x.shape[1]
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    gammas = kernel(
        [
            trial.suggest_float(f"gamma_{j}", 1e-5, 1e2, log=True)
            for j in range(n_features)
        ]
    )
    model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("svc", SVC(kernel=gammas, C=C)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring="balanced_accuracy")
    return scores.mean()


study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(
    lambda trial: objective(trial, X, W), n_trials=args.n_trials, show_progress_bar=True
)

best = study.best_params
best_gammas = kernel([best[f"gamma_{j}"] for j in range(X.shape[1])])
model = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("svc", SVC(kernel=best_gammas, C=best["C"])),
    ]
)
model.fit(X, W)

with open(args.out, "wb") as f:
    pickle.dump(model, f)
