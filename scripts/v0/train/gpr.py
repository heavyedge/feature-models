import argparse
import logging
import pathlib
import sys

import gpytorch
import numpy as np
import optuna
import pandas as pd
import torch
import v0.model.gpr as model_module  # Needs PYTHONPATH=scripts
import v0.model.load as load_module  # Needs PYTHONPATH=scripts
import v0.model.scale as scaler_module  # Needs PYTHONPATH=scripts
from gpytorch.constraints import Positive
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from save import save_gpr  # Needs PYTHONPATH=scripts

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
model_group = parser.add_argument_group("input data and model")
training_group = parser.add_argument_group("per-trial training")
hpo_group = parser.add_argument_group("hyperparameter optimization")

model_group.add_argument("Xtrain", type=pathlib.Path, help="Training feature csv file.")
model_group.add_argument("ytrain", type=pathlib.Path, help="Training target csv file.")
model_group.add_argument(
    "Xval", type=pathlib.Path, nargs="?", help="Validation feature csv file."
)
model_group.add_argument(
    "yval", type=pathlib.Path, nargs="?", help="Validation target csv file."
)
model_group.add_argument(
    "prior_mean",
    type=pathlib.Path,
    help="Prior mean model weight file.",
)
parser.add_argument(
    "--index-col", type=int, nargs="*", help="Index columns for X and y."
)
model_group.add_argument("--target", type=str, help="Target variable name.")
model_group.add_argument("--model", type=str, help="Model name.")
model_group.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
model_group.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")

training_group.add_argument("--num-epochs", type=int, help="Number of maximum epochs.")
training_group.add_argument(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Initial learning rate for optimizer.",
)
training_group.add_argument(
    "--early-stopping-patience-ratio",
    type=float,
    default=0.1,
    help=(
        "Fraction of maximum epochs without validation-loss improvement before "
        "early stopping."
    ),
)
training_group.add_argument(
    "--early-stopping-min-delta",
    type=float,
    default=0.0,
    help="Minimum validation-loss decrease required to reset early stopping.",
)
training_group.add_argument(
    "--lr-scheduler-patience",
    type=int,
    default=10,
    help="Epochs without validation-loss improvement before reducing learning rate.",
)
training_group.add_argument(
    "--lr-scheduler-factor",
    type=float,
    default=0.5,
    help="Factor by which to reduce the learning rate.",
)
training_group.add_argument(
    "--min-learning-rate",
    type=float,
    default=1e-6,
    help="Minimum learning rate for the scheduler.",
)
hpo_group.add_argument(
    "--n-trials",
    type=int,
    default=50,
    help="Number of trials for hyperparameter optimization.",
)
hpo_group.add_argument(
    "--pruning-patience-ratio",
    type=float,
    default=0.2,
    help=(
        "Fraction of maximum epochs without validation-score improvement before "
        "allowing trial pruning."
    ),
)
hpo_group.add_argument(
    "--prior-loc-min",
    type=float,
    default=-10.0,
    help="Smallest log-space location for the LogNormal prior.",
)
hpo_group.add_argument(
    "--prior-loc-max",
    type=float,
    default=2.0,
    help="Largest log-space location for the LogNormal prior.",
)
hpo_group.add_argument(
    "--prior-scale-min",
    type=float,
    default=0.1,
    help="Smallest log-space scale for the LogNormal prior.",
)
hpo_group.add_argument(
    "--prior-scale-max",
    type=float,
    default=2.0,
    help="Largest log-space scale for the LogNormal prior.",
)
hpo_group.add_argument(
    "--storage",
    type=str,
    default=None,
    help="Optuna storage URL.",
)
hpo_group.add_argument(
    "--study-name",
    type=str,
    default=None,
    help="Optuna study name.",
)
args = parser.parse_args()

has_validation = args.Xval is not None or args.yval is not None
if (args.Xval is None) != (args.yval is None):
    parser.error("Xval and yval must be provided together.")
if has_validation and args.num_epochs is None:
    parser.error("--num-epochs is required when validation data is provided.")
if not has_validation and args.storage is None:
    parser.error("--storage is required when validation data is not provided.")

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

if has_validation:
    early_stopping_patience = max(
        1, round(args.num_epochs * args.early_stopping_patience_ratio)
    )
    pruning_patience = max(1, round(args.num_epochs * args.pruning_patience_ratio))

Xtrain_df = pd.read_csv(args.Xtrain, index_col=args.index_col)
Xtrain_arr = np.stack(
    [Xtrain_df.loc[fold] for fold in sorted(Xtrain_df.index.unique())], axis=0
)
Xtrain = torch.tensor(Xtrain_arr).float().to(device)  # (*K, N, D)

ytrain_df = pd.read_csv(args.ytrain, index_col=args.index_col)[args.target]
ytrain_arr = np.stack(
    [ytrain_df.loc[fold] for fold in sorted(ytrain_df.index.unique())], axis=0
)
ytrain = torch.tensor(ytrain_arr).float().to(device)  # (*K, N)

if has_validation:
    Xval_df = pd.read_csv(args.Xval, index_col=args.index_col)
    Xval_arr = np.stack(
        [Xval_df.loc[fold] for fold in sorted(Xval_df.index.unique())], axis=0
    )
    Xval = torch.tensor(Xval_arr).float().to(device)  # (*K, N, D)
    yval_df = pd.read_csv(args.yval, index_col=args.index_col)[args.target]
    yval_arr = np.stack(
        [yval_df.loc[fold] for fold in sorted(yval_df.index.unique())], axis=0
    )
    yval = torch.tensor(yval_arr).float().to(device)  # (*K, N)

priormean_loader = getattr(load_module, "load_PriorMean_" + args.target)
mean = priormean_loader(path=args.prior_mean, device=device)
mean.eval()

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
batch_shape = Xtrain.shape[:-2]

model_class = getattr(model_module, args.model)


def train_with_priors(noise_prior, lengthscale_prior, trial=None):
    """Train one GP trial and return its best batch-mean validation loss."""
    torch.manual_seed(42)
    trial_label = trial.number if trial is not None else "best"

    X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
    y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)
    X_scaler.train()
    Xtrain_scaled = X_scaler(Xtrain)

    likelihood = GaussianLikelihood(
        batch_shape=batch_shape,
        noise_prior=noise_prior,
        noise_constraint=Positive(),
    ).to(device)
    with torch.no_grad():
        y_scaler.train()
        res = y_scaler((ytrain - mean(Xtrain)).unsqueeze(-1)).squeeze(-1)
    model = model_class(
        Xtrain_scaled,
        res,
        likelihood,
        batch_shape=batch_shape,
        lengthscale_prior_loc=lengthscale_prior.loc,
        lengthscale_prior_scale=lengthscale_prior.scale,
    ).to(device)

    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(
        list(X_scaler.parameters())
        + list(y_scaler.parameters())
        + list(model.parameters()),
        lr=args.learning_rate,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_learning_rate,
    )
    best_val_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0
    training_loss_history = []

    for epoch in range(args.num_epochs):
        X_scaler.train()
        y_scaler.train()
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        Xtrain_scaled = X_scaler(Xtrain)
        with torch.no_grad():
            train_mean = mean(Xtrain)
        res = y_scaler((ytrain - train_mean).unsqueeze(-1)).squeeze(-1)
        model.set_train_data(
            inputs=Xtrain_scaled.detach(), targets=res.detach(), strict=False
        )

        output = model(Xtrain_scaled)
        # ExactMarginalLogLikelihood returns one value per batched fold.
        train_loss = -mll(output, res).mean()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_scaler.eval()
            y_scaler.eval()
            likelihood.eval()
            model.eval()
            Xval_scaled = X_scaler(Xval)
            val_res = y_scaler((yval - mean(Xval)).unsqueeze(-1)).squeeze(-1)
            val_log_prob = likelihood.expected_log_prob(val_res, model(Xval_scaled))
            # Average observations within each fold, then average the folds so
            # Optuna receives one fold-balanced scalar score.
            val_loss = -val_log_prob.mean(dim=-1).mean()

        current_val_loss = val_loss.item()
        training_loss_history.append(train_loss.item())
        lr_scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if trial is not None:
            # Intermediate values are persisted in Optuna storage and power
            # both pruning and the native intermediate-values visualization.
            trial.report(current_val_loss, step=epoch)
            if trial.should_prune():
                trial.set_user_attr("training_loss_history", training_loss_history)
                trial.set_user_attr("best_epoch", best_epoch)
                raise optuna.TrialPruned()

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Trial %s, epoch %d: train loss %.6f, validation loss %.6f, noise %.2e",
                trial_label,
                epoch + 1,
                train_loss.item(),
                current_val_loss,
                likelihood.noise.mean().item(),
            )

        if epochs_without_improvement >= early_stopping_patience:
            break

    if trial is not None:
        trial.set_user_attr("training_loss_history", training_loss_history)
        trial.set_user_attr("best_epoch", best_epoch)
    return best_val_loss


def objective(trial):
    noise_prior_loc = trial.suggest_float(
        "noise_prior_loc",
        args.prior_loc_min,
        args.prior_loc_max,
    )
    noise_prior_scale = trial.suggest_float(
        "noise_prior_scale",
        args.prior_scale_min,
        args.prior_scale_max,
        log=True,
    )
    lengthscale_prior_loc = trial.suggest_float(
        "lengthscale_prior_loc",
        args.prior_loc_min,
        args.prior_loc_max,
    )
    lengthscale_prior_scale = trial.suggest_float(
        "lengthscale_prior_scale",
        args.prior_scale_min,
        args.prior_scale_max,
        log=True,
    )
    val_loss = train_with_priors(
        LogNormalPrior(noise_prior_loc, noise_prior_scale),
        LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale),
        trial,
    )
    return val_loss


def train_on_all_data(noise_prior, lengthscale_prior, num_epochs):
    """Fit the selected configuration on all provided data for fixed epochs."""
    torch.manual_seed(42)
    if has_validation:
        Xall = torch.cat((Xtrain, Xval), dim=-2)
        yall = torch.cat((ytrain, yval), dim=-1)
    else:
        Xall = Xtrain
        yall = ytrain

    X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
    y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)
    likelihood = GaussianLikelihood(
        batch_shape=batch_shape,
        noise_prior=noise_prior,
        noise_constraint=Positive(),
    ).to(device)

    X_scaler.train()
    Xall_scaled = X_scaler(Xall)
    with torch.no_grad():
        y_scaler.train()
        res = y_scaler((yall - mean(Xall)).unsqueeze(-1)).squeeze(-1)
    model = model_class(
        Xall_scaled,
        res,
        likelihood,
        batch_shape=batch_shape,
        lengthscale_prior_loc=lengthscale_prior.loc,
        lengthscale_prior_scale=lengthscale_prior.scale,
    ).to(device)

    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(
        list(X_scaler.parameters())
        + list(y_scaler.parameters())
        + list(model.parameters()),
        lr=args.learning_rate,
    )
    for epoch in range(num_epochs):
        X_scaler.train()
        y_scaler.train()
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        Xall_scaled = X_scaler(Xall)
        with torch.no_grad():
            all_mean = mean(Xall)
        res = y_scaler((yall - all_mean).unsqueeze(-1)).squeeze(-1)
        model.set_train_data(
            inputs=Xall_scaled.detach(), targets=res.detach(), strict=False
        )
        loss = -mll(model(Xall_scaled), res).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Final training, epoch %d/%d: loss %.6f",
                epoch + 1,
                num_epochs,
                loss.item(),
            )

    return Xall, yall, X_scaler, y_scaler, likelihood, model


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = args.study_name if args.study_name is not None else args.out.stem
if has_validation:
    median_pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=pruning_patience,
        interval_steps=max(1, pruning_patience // 10),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.PatientPruner(
            median_pruner,
            patience=pruning_patience,
        ),
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
else:
    # The final-data path must be read-only with respect to HPO: load the
    # completed study and train once using its selected configuration.
    study = optuna.load_study(study_name=study_name, storage=args.storage)

best_trial = study.best_trial
best_noise_prior_loc = best_trial.params["noise_prior_loc"]
best_noise_prior_scale = best_trial.params["noise_prior_scale"]
best_lengthscale_prior_loc = best_trial.params["lengthscale_prior_loc"]
best_lengthscale_prior_scale = best_trial.params["lengthscale_prior_scale"]
logger.info(
    "Best priors: noise=LogNormal(loc=%.6g, scale=%.6g), "
    "lengthscale=LogNormal(loc=%.6g, scale=%.6g) "
    "(validation loss: %.6f)",
    best_noise_prior_loc,
    best_noise_prior_scale,
    best_lengthscale_prior_loc,
    best_lengthscale_prior_scale,
    best_trial.value,
)

if "best_epoch" not in best_trial.user_attrs:
    raise RuntimeError(
        f"Best trial {best_trial.number} in study {study_name!r} has no "
        "'best_epoch' user attribute."
    )
best_epoch = max(1, int(best_trial.user_attrs["best_epoch"]))
if has_validation:
    best_epoch = min(best_epoch, args.num_epochs)
logger.info("Final training on all provided data for %d epochs.", best_epoch)

# Refit the selected configuration on all available labelled data.  The epoch
# count is fixed from the best trial because no validation set remains here.
(
    Xall,
    yall,
    X_scaler,
    y_scaler,
    likelihood,
    model,
) = train_on_all_data(
    LogNormalPrior(best_noise_prior_loc, best_noise_prior_scale),
    LogNormalPrior(best_lengthscale_prior_loc, best_lengthscale_prior_scale),
    best_epoch,
)

save_gpr(
    Xall,
    yall,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    args.out,
)
