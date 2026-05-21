import argparse
import pathlib
import pickle

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class AnisotropicMatern52:
    """Matern 5/2 kernel with per-feature bandwidth:
    K(x,x') = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r),
    where r = sqrt(sum_j gamma_j*(x_j-x'_j)^2).
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas)

    def __call__(self, X, Y):
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        r2 = np.einsum("nmd,d->nm", diff**2, self.gammas)
        r = np.sqrt(np.maximum(r2, 0.0))
        sqrt5_r = np.sqrt(5.0) * r
        return (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * np.exp(-sqrt5_r)


parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Observed X csv file.")
parser.add_argument("window", type=pathlib.Path, help="Quality window npy file.")
parser.add_argument("--n-trials", type=int, help="Number of Optuna trials.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output pkl file.")
args = parser.parse_args()

X = pd.read_csv(args.X).to_numpy()
W = np.load(args.window)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, x, y):
    n_features = x.shape[1]
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    gammas = AnisotropicMatern52(
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
best_gammas = AnisotropicMatern52([best[f"gamma_{j}"] for j in range(X.shape[1])])
model = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("svc", SVC(kernel=best_gammas, C=best["C"])),
    ]
)
model.fit(X, W)

with open(args.out, "wb") as f:
    pickle.dump(model, f)
