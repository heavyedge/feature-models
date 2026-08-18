import argparse
import logging
import pathlib
import sys

import gpytorch
import numpy as np
import optuna
import torch
from gpytorch.mlls import VariationalELBO
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP

from models.v0.feature_models import gpqr as model_module
from models.v0.feature_models import load as load_module
from models.v0.feature_models import scale as scaler_module
from models.v0.feature_models.likelihoods import (
    CenterGapQuantilesLikelihood,
    DirectQuantilesLikelihood,
)
from scripts.v0.train.batch import load_batched_arrays
from scripts.v0.train.inducing import unique_inducing_points_per_fold
from scripts.v0.train.save import save_gpqr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

MAX_DEFAULT_LR_SCHEDULER_PATIENCE = 50
BASELINE_PRIOR_PARAMS = {
    "noise_prior_loc": -4.0,
    "noise_prior_scale": 0.5,
    "lengthscale_prior_loc": -1.0,
    "lengthscale_prior_scale": 0.5,
}
DEFAULT_NUM_LATENTS = 3

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
parser.add_argument(
    "--batch-col",
    type=int,
    nargs="*",
    help="X CSV column(s) defining batch dimensions.",
)
model_group.add_argument("--target", type=str, help="Target variable name.")
model_group.add_argument("--model", type=str, help="Model name.")
model_group.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    help="Quantiles for the model.",
)
model_group.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
model_group.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")

training_group.add_argument("--num-epochs", type=int, help="Number of maximum epochs.")
training_group.add_argument(
    "--num-likelihood-samples",
    type=int,
    default=10,
    help=(
        "Number of latent GP samples used to estimate expected log likelihoods "
        "during training and validation."
    ),
)
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
    default=1e-4,
    help="Minimum validation-loss decrease required to reset early stopping.",
)
training_group.add_argument(
    "--lr-scheduler-patience-ratio",
    type=float,
    default=0.02,
    help=(
        "Fraction of maximum epochs used as scheduler patience when absolute "
        "patience is omitted (capped at 50 epochs)."
    ),
)
training_group.add_argument(
    "--lr-scheduler-factor",
    type=float,
    default=0.3,
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
    default=100,
    help="Number of trials for hyperparameter optimization.",
)
hpo_group.add_argument(
    "--n-startup-trials",
    type=int,
    default=10,
    help="Completed random trials before TPE sampling and median pruning begin.",
)
hpo_group.add_argument(
    "--pruning-warmup-ratio",
    type=float,
    default=0.05,
    help="Fraction of maximum epochs to run before median pruning begins.",
)
hpo_group.add_argument(
    "--pruning-patience-ratio",
    type=float,
    default=0.02,
    help=(
        "Fraction of maximum epochs without sufficient validation improvement "
        "required before accepting a median-pruner decision."
    ),
)
hpo_group.add_argument(
    "--noise-prior-loc-min",
    type=float,
    default=-8.0,
    help="Smallest log-space location for the noise LogNormal prior.",
)
hpo_group.add_argument(
    "--noise-prior-loc-max",
    type=float,
    default=-1.0,
    help="Largest log-space location for the noise LogNormal prior.",
)
hpo_group.add_argument(
    "--lengthscale-prior-loc-min",
    type=float,
    default=-3.0,
    help="Smallest log-space location for the lengthscale LogNormal prior.",
)
hpo_group.add_argument(
    "--lengthscale-prior-loc-max",
    type=float,
    default=1.0,
    help="Largest log-space location for the lengthscale LogNormal prior.",
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
    default=1.5,
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
if has_validation and args.num_epochs <= 0:
    parser.error("--num-epochs must be positive.")
if args.num_likelihood_samples <= 0:
    parser.error("--num-likelihood-samples must be positive.")
if args.learning_rate <= 0:
    parser.error("--learning-rate must be positive.")
if args.min_learning_rate < 0 or args.min_learning_rate > args.learning_rate:
    parser.error("--min-learning-rate must be between 0 and --learning-rate.")
if not 0 < args.lr_scheduler_factor < 1:
    parser.error("--lr-scheduler-factor must be between 0 and 1.")
if not 0 < args.lr_scheduler_patience_ratio <= 1:
    parser.error("--lr-scheduler-patience-ratio must be in (0, 1].")
if not 0 < args.early_stopping_patience_ratio <= 1:
    parser.error("--early-stopping-patience-ratio must be in (0, 1].")
if args.early_stopping_min_delta < 0:
    parser.error("--early-stopping-min-delta cannot be negative.")
if args.n_trials <= 0:
    parser.error("--n-trials must be positive.")
if args.n_startup_trials < 0:
    parser.error("--n-startup-trials cannot be negative.")
if len(args.quantiles) < 2:
    parser.error("--quantiles must contain at least two values.")
if not all(np.isfinite(q) and 0 < q < 1 for q in args.quantiles):
    parser.error("--quantiles values must be finite and in (0, 1).")
if any(left >= right for left, right in zip(args.quantiles, args.quantiles[1:])):
    parser.error("--quantiles values must be strictly increasing.")
if not 0 <= args.pruning_warmup_ratio <= 1:
    parser.error("--pruning-warmup-ratio must be in [0, 1].")
if not 0 <= args.pruning_patience_ratio <= 1:
    parser.error("--pruning-patience-ratio must be in [0, 1].")
if args.prior_scale_min <= 0 or args.prior_scale_min >= args.prior_scale_max:
    parser.error("Prior scale bounds must be positive and strictly increasing.")

if args.noise_prior_loc_min >= args.noise_prior_loc_max:
    parser.error("Noise-prior location bounds must be strictly increasing.")
if args.lengthscale_prior_loc_min >= args.lengthscale_prior_loc_max:
    parser.error("Lengthscale-prior location bounds must be strictly increasing.")

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

if has_validation:
    early_stopping_patience = max(
        1, round(args.num_epochs * args.early_stopping_patience_ratio)
    )
    lr_scheduler_patience = min(
        MAX_DEFAULT_LR_SCHEDULER_PATIENCE,
        max(1, round(args.num_epochs * args.lr_scheduler_patience_ratio)),
    )
    pruning_warmup = max(0, round(args.num_epochs * args.pruning_warmup_ratio))
    pruning_patience = max(1, round(args.num_epochs * args.pruning_patience_ratio))
    pruning_interval = max(1, min(25, pruning_patience // 2))
    logger.info(
        "Training controls: scheduler patience=%d, early-stopping patience=%d, "
        "pruning warmup=%d, pruning patience=%d epochs.",
        lr_scheduler_patience,
        early_stopping_patience,
        pruning_warmup,
        pruning_patience,
    )

try:
    Xtrain_arr, ytrain_arr = load_batched_arrays(
        args.Xtrain, args.ytrain, args.target, args.index_col, args.batch_col
    )
except ValueError as exc:
    parser.error(str(exc))
Xtrain = torch.tensor(Xtrain_arr).float().to(device)  # (*B, N, D)
ytrain = torch.tensor(ytrain_arr).float().to(device)  # (*B, N)

priormean_loader = getattr(load_module, "load_PriorMean_" + args.target)
mean = priormean_loader(path=args.prior_mean, device=device)
mean.eval()
with torch.no_grad():
    train_mean = mean(Xtrain)
res_train = ytrain - train_mean

if has_validation:
    try:
        Xval_arr, yval_arr = load_batched_arrays(
            args.Xval, args.yval, args.target, args.index_col, args.batch_col
        )
    except ValueError as exc:
        parser.error(str(exc))
    Xval = torch.tensor(Xval_arr).float().to(device)  # (*B, N, D)
    yval = torch.tensor(yval_arr).float().to(device)  # (*B, N)
    if Xval.shape[:-2] != Xtrain.shape[:-2]:
        parser.error("Training and validation data must have the same batch shape.")

    with torch.no_grad():
        val_mean = mean(Xval)
    res_val = yval - val_mean

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
batch_shape = Xtrain.shape[:-2]

X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
X_scaler.train()
with torch.no_grad():
    Xtrain_scaled = X_scaler(Xtrain)
X_scaler.eval()

y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)
y_scaler.train()
with torch.no_grad():
    res_train_scaled = y_scaler(res_train.unsqueeze(-1)).squeeze(-1)
y_scaler.eval()

if has_validation:
    with torch.no_grad():
        Xval_scaled = X_scaler(Xval)
        res_val_scaled = y_scaler(res_val.unsqueeze(-1)).squeeze(-1)

quantiles = torch.tensor(args.quantiles, dtype=torch.float32, device=device)
central_quantile_idx = np.argmin(np.abs(quantiles.detach().cpu().numpy() - 0.5))
num_quantiles = len(quantiles)
num_lower_quantiles = central_quantile_idx

model_class = getattr(model_module, args.model)

if issubclass(model_class, CenterGapQuantileGP):
    likelihood_class = CenterGapQuantilesLikelihood
elif issubclass(model_class, DirectQuantileGP):
    likelihood_class = DirectQuantilesLikelihood
else:
    raise ValueError(f"Unknown model class: {model_class}")


# Keep the full training set for the ELBO; only the variational inducing set
# is deduplicated.
Xtrain_inducing_points = unique_inducing_points_per_fold(Xtrain_scaled)


def train_with_hyperparameters(
    noise_prior_loc,
    noise_prior_scale,
    lengthscale_prior_loc,
    lengthscale_prior_scale,
    num_latents,
    trial=None,
):
    """Train one GPQR trial and return its best batch-mean validation loss."""
    torch.manual_seed(42)
    trial_label = trial.number if trial is not None else "best"

    likelihood = likelihood_class(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=batch_shape,
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
    ).to(device)
    model = model_class(
        # Each HPO trial must start from the same immutable inducing set.
        inducing_points=Xtrain_inducing_points.clone().detach(),
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_latents,
        batch_shape=batch_shape,
        lengthscale_prior_loc=lengthscale_prior_loc,
        lengthscale_prior_scale=lengthscale_prior_scale,
    ).to(device)

    mll = VariationalELBO(likelihood, model, num_data=num_data)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=args.learning_rate,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=args.early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=args.min_learning_rate,
    )
    best_val_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0
    training_loss_history = []
    lr_reductions = []

    def save_trial_progress(stop_reason):
        if trial is None:
            return
        trial.set_user_attr("training_loss_history", training_loss_history)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("epochs_trained", len(training_loss_history))
        trial.set_user_attr("lr_reductions", lr_reductions)
        trial.set_user_attr("stop_reason", stop_reason)

    for epoch in range(args.num_epochs):
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        output = model(Xtrain_scaled)
        # VariationalELBO returns values over the model's batched folds.
        with gpytorch.settings.num_likelihood_samples(args.num_likelihood_samples):
            train_loss = -mll(output, res_train_scaled).mean()
        if not torch.isfinite(train_loss):
            save_trial_progress("non_finite_train_loss")
            if trial is not None:
                raise optuna.TrialPruned("Non-finite training loss.")
            raise RuntimeError("Non-finite training loss during final fit.")
        train_loss.backward()
        optimizer.step()

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.num_likelihood_samples(args.num_likelihood_samples),
        ):
            likelihood.eval()
            model.eval()
            val_log_prob = likelihood.expected_log_prob(
                res_val_scaled, model(Xval_scaled)
            )
            # Average observations and quantiles within each fold, then average
            # the folds so Optuna receives one fold-balanced scalar score.
            val_loss = -val_log_prob.mean(dim=(-2, -1)).mean()

        current_val_loss = val_loss.item()
        training_loss_history.append(train_loss.item())
        if not np.isfinite(current_val_loss):
            save_trial_progress("non_finite_validation_loss")
            if trial is not None:
                raise optuna.TrialPruned("Non-finite validation loss.")
            raise RuntimeError("Non-finite validation loss.")

        previous_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(current_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr:
            # Record the completed epoch after which the new rate takes effect.
            lr_reductions.append([epoch + 1, current_lr])

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
                save_trial_progress("pruned")
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
            stop_reason = "early_stopping"
            break
    else:
        stop_reason = "max_epochs"

    save_trial_progress(stop_reason)
    return best_val_loss


def objective(trial):
    noise_prior_loc = trial.suggest_float(
        "noise_prior_loc",
        args.noise_prior_loc_min,
        args.noise_prior_loc_max,
    )
    noise_prior_scale = trial.suggest_float(
        "noise_prior_scale",
        args.prior_scale_min,
        args.prior_scale_max,
        log=True,
    )
    lengthscale_prior_loc = trial.suggest_float(
        "lengthscale_prior_loc",
        args.lengthscale_prior_loc_min,
        args.lengthscale_prior_loc_max,
    )
    lengthscale_prior_scale = trial.suggest_float(
        "lengthscale_prior_scale",
        args.prior_scale_min,
        args.prior_scale_max,
        log=True,
    )
    num_latents = trial.suggest_int("num_latents", 2, num_quantiles)
    val_loss = train_with_hyperparameters(
        noise_prior_loc,
        noise_prior_scale,
        lengthscale_prior_loc,
        lengthscale_prior_scale,
        num_latents,
        trial,
    )
    return val_loss


def train_on_all_data(
    noise_prior_loc,
    noise_prior_scale,
    lengthscale_prior_loc,
    lengthscale_prior_scale,
    num_latents,
    num_epochs,
    lr_reductions,
):
    """Fit on all data while replaying the selected trial's learning-rate path."""
    torch.manual_seed(42)
    if has_validation:
        Xall = torch.cat((Xtrain, Xval), dim=-2)
        yall = torch.cat((ytrain, yval), dim=-1)
    else:
        Xall = Xtrain
        yall = ytrain

    likelihood = likelihood_class(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=batch_shape,
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
    ).to(device)

    X_scaler.train()
    with torch.no_grad():
        Xall_scaled = X_scaler(Xall)
    X_scaler.eval()

    y_scaler.train()
    with torch.no_grad():
        res_all_scaled = y_scaler((yall - mean(Xall)).unsqueeze(-1)).squeeze(-1)
    y_scaler.eval()

    inducing_points = unique_inducing_points_per_fold(Xall_scaled)
    model = model_class(
        inducing_points=inducing_points,
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_latents,
        batch_shape=batch_shape,
        lengthscale_prior_loc=lengthscale_prior_loc,
        lengthscale_prior_scale=lengthscale_prior_scale,
    ).to(device)

    mll = VariationalELBO(likelihood, model, num_data=Xall.shape[-2])
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=args.learning_rate,
    )
    lr_reductions_by_epoch = {
        int(completed_epoch): float(new_lr) for completed_epoch, new_lr in lr_reductions
    }
    for epoch in range(num_epochs):
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        with gpytorch.settings.num_likelihood_samples(args.num_likelihood_samples):
            loss = -mll(model(Xall_scaled), res_all_scaled).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss during final fit.")
        loss.backward()
        optimizer.step()

        completed_epoch = epoch + 1
        if completed_epoch in lr_reductions_by_epoch:
            new_lr = lr_reductions_by_epoch[completed_epoch]
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = new_lr

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Final training, epoch %d/%d: loss %.6f",
                epoch + 1,
                num_epochs,
                loss.item(),
            )

    return X_scaler, y_scaler, likelihood, model


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = args.study_name if args.study_name is not None else args.out.stem
if has_validation:
    median_pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=pruning_warmup,
        interval_steps=pruning_interval,
        n_min_trials=3,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=args.n_startup_trials,
            n_ei_candidates=48,
            multivariate=True,
        ),
        pruner=optuna.pruners.PatientPruner(
            median_pruner,
            patience=pruning_patience,
            min_delta=args.early_stopping_min_delta,
        ),
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    if not study.trials:
        baseline_params = {
            "noise_prior_loc": float(
                np.clip(
                    BASELINE_PRIOR_PARAMS["noise_prior_loc"],
                    args.noise_prior_loc_min,
                    args.noise_prior_loc_max,
                )
            ),
            "noise_prior_scale": float(
                np.clip(
                    BASELINE_PRIOR_PARAMS["noise_prior_scale"],
                    args.prior_scale_min,
                    args.prior_scale_max,
                )
            ),
            "lengthscale_prior_loc": float(
                np.clip(
                    BASELINE_PRIOR_PARAMS["lengthscale_prior_loc"],
                    args.lengthscale_prior_loc_min,
                    args.lengthscale_prior_loc_max,
                )
            ),
            "lengthscale_prior_scale": float(
                np.clip(
                    BASELINE_PRIOR_PARAMS["lengthscale_prior_scale"],
                    args.prior_scale_min,
                    args.prior_scale_max,
                )
            ),
            "num_latents": int(np.clip(DEFAULT_NUM_LATENTS, 2, num_quantiles)),
        }
        study.enqueue_trial(baseline_params)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(torch.linalg.LinAlgError,),
    )
else:
    # The final-data path must be read-only with respect to HPO: load the
    # completed study and train once using its selected configuration.
    study = optuna.load_study(study_name=study_name, storage=args.storage)

best_trial = study.best_trial
best_noise_prior_loc = best_trial.params["noise_prior_loc"]
best_noise_prior_scale = best_trial.params["noise_prior_scale"]
best_lengthscale_prior_loc = best_trial.params["lengthscale_prior_loc"]
best_lengthscale_prior_scale = best_trial.params["lengthscale_prior_scale"]
best_num_latents = best_trial.params["num_latents"]
logger.info(
    "Best priors: noise=LogNormal(loc=%.6g, scale=%.6g), "
    "lengthscale=LogNormal(loc=%.6g, scale=%.6g); num_latents=%d "
    "(validation loss: %.6f)",
    best_noise_prior_loc,
    best_noise_prior_scale,
    best_lengthscale_prior_loc,
    best_lengthscale_prior_scale,
    best_num_latents,
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
best_lr_reductions = best_trial.user_attrs.get("lr_reductions", [])
logger.info(
    "Final training on all provided data for %d epochs with %d learning-rate "
    "reductions replayed.",
    best_epoch,
    sum(int(completed_epoch) < best_epoch for completed_epoch, _ in best_lr_reductions),
)

# Refit the selected configuration on all available labelled data.  The epoch
# count is fixed from the best trial because no validation set remains here.
(
    X_scaler,
    y_scaler,
    likelihood,
    model,
) = train_on_all_data(
    best_noise_prior_loc,
    best_noise_prior_scale,
    best_lengthscale_prior_loc,
    best_lengthscale_prior_scale,
    best_num_latents,
    best_epoch,
    best_lr_reductions,
)

save_gpqr(
    quantiles,
    X_scaler,
    y_scaler,
    likelihood,
    model,
    args.out,
)
