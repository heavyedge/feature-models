import argparse
import copy
import logging
import pathlib
import sys

import gpytorch
import optuna
import pandas as pd
import torch
import v0.model.prior as prior_module  # Needs PYTHONPATH=scripts
import v0.model.scale as scaler_module  # Needs PYTHONPATH=scripts
import v1.model.gpr as model_module  # Needs PYTHONPATH=scripts
from gpytorch.constraints import Positive
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from v1.train.save import save_gpr  # Needs PYTHONPATH=scripts

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
model_group.add_argument("Xval", type=pathlib.Path, help="Validation feature csv file.")
model_group.add_argument("yval", type=pathlib.Path, help="Validation target csv file.")
model_group.add_argument(
    "prior_mean",
    type=pathlib.Path,
    help="Prior mean model weight file.",
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
    default=0.02,
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
    default=0.02,
    help="Fraction of maximum epochs to wait before enabling trial pruning.",
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
args = parser.parse_args()

early_stopping_patience = max(
    1, round(args.num_epochs * args.early_stopping_patience_ratio)
)
pruning_patience = max(1, round(args.num_epochs * args.pruning_patience_ratio))

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

Xtrain = torch.tensor(pd.read_csv(args.Xtrain).values).float().to(device)
ytrain = torch.tensor(pd.read_csv(args.ytrain)[args.target].values).float().to(device)
Xval = torch.tensor(pd.read_csv(args.Xval).values).float().to(device)
yval = torch.tensor(pd.read_csv(args.yval)[args.target].values).float().to(device)

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
batch_shape = Xtrain.shape[:-2]

mean_class = getattr(prior_module, "PriorMean_" + args.target)
mean = mean_class(batch_shape=batch_shape).to(device)
mean.load_state_dict(torch.load(args.prior_mean, map_location=device))
mean.eval()

model_class = getattr(model_module, args.model)


def train_with_priors(noise_prior, lengthscale_prior, trial=None):
    """Train one GP trial and return its best validation-loss checkpoint."""
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
        lengthscale_prior=lengthscale_prior,
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
    best_state = None
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
        train_loss = -mll(output, res)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_scaler.eval()
            y_scaler.eval()
            likelihood.eval()
            model.eval()
            Xval_scaled = X_scaler(Xval)
            val_res = y_scaler((yval - mean(Xval)).unsqueeze(-1)).squeeze(-1)
            val_loss = -likelihood.expected_log_prob(val_res, model(Xval_scaled)).mean()

        current_val_loss = val_loss.item()
        training_loss_history.append(train_loss.item())
        lr_scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = current_val_loss
            best_state = {
                "X_scaler": copy.deepcopy(X_scaler.state_dict()),
                "y_scaler": copy.deepcopy(y_scaler.state_dict()),
                "likelihood": copy.deepcopy(likelihood.state_dict()),
                "model": copy.deepcopy(model.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if trial is not None:
            # Intermediate values are persisted in Optuna storage and power
            # both pruning and the native intermediate-values visualization.
            trial.report(current_val_loss, step=epoch)
            if trial.should_prune():
                trial.set_user_attr("training_loss_history", training_loss_history)
                trial.set_user_attr(
                    "best_epoch", epoch + 1 - epochs_without_improvement
                )
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
        trial.set_user_attr("best_epoch", epoch + 1 - epochs_without_improvement)
    return best_val_loss, best_state


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
    val_loss, _ = train_with_priors(
        LogNormalPrior(noise_prior_loc, noise_prior_scale),
        LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale),
        trial,
    )
    return val_loss


def train_on_all_data(noise_prior, lengthscale_prior, num_epochs):
    """Fit the selected configuration on train+validation for a fixed epoch count."""
    torch.manual_seed(42)
    Xall = torch.cat((Xtrain, Xval), dim=-2)
    yall = torch.cat((ytrain, yval), dim=-1)

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
        lengthscale_prior=lengthscale_prior,
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
        loss = -mll(model(Xall_scaled), res)
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
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=pruning_patience,
        interval_steps=max(1, pruning_patience // 10),
    ),
    study_name=f"{args.out.stem}",
    storage=args.storage if args.storage is not None else None,
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.n_trials)
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

best_epoch = best_trial.user_attrs.get("best_epoch", args.num_epochs)
best_epoch = max(1, min(int(best_epoch), args.num_epochs))
logger.info("Final training on train+validation data for %d epochs.", best_epoch)

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
