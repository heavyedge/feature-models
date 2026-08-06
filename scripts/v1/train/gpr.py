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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from v1.train.save import save_gpr  # Needs PYTHONPATH=scripts

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("Xtrain", type=pathlib.Path, help="Training feature csv file.")
parser.add_argument("ytrain", type=pathlib.Path, help="Training target csv file.")
parser.add_argument("Xval", type=pathlib.Path, help="Validation feature csv file.")
parser.add_argument("yval", type=pathlib.Path, help="Validation target csv file.")
parser.add_argument(
    "prior_mean",
    type=pathlib.Path,
    help="Prior mean model weight file.",
)
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument("--num-epochs", type=int, help="Number of maximum epochs.")
parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Initial learning rate for optimizer.",
)
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=20,
    help="Stop after this many epochs without validation-loss improvement.",
)
parser.add_argument(
    "--early-stopping-min-delta",
    type=float,
    default=0.0,
    help="Minimum validation-loss decrease required to reset early stopping.",
)
parser.add_argument(
    "--lr-scheduler-patience",
    type=int,
    default=10,
    help="Epochs without validation-loss improvement before reducing learning rate.",
)
parser.add_argument(
    "--lr-scheduler-factor",
    type=float,
    default=0.5,
    help="Factor by which to reduce the learning rate.",
)
parser.add_argument(
    "--min-learning-rate",
    type=float,
    default=1e-6,
    help="Minimum learning rate for the scheduler.",
)
parser.add_argument(
    "--n-trials",
    type=int,
    default=50,
    help="Number of trials for hyperparameter optimization.",
)
parser.add_argument(
    "--lengthscale-lower-bound-min",
    type=float,
    default=0.01,
    help="Smallest per-dimension ARD lengthscale lower bound considered by Optuna.",
)
parser.add_argument(
    "--lengthscale-lower-bound-max",
    type=float,
    default=1.0,
    help="Largest per-dimension ARD lengthscale lower bound considered by Optuna.",
)
parser.add_argument(
    "--storage-name",
    type=str,
    default=None,
    help="Optuna storage name for resuming trials.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.n_trials < 1:
    parser.error("--n-trials must be at least 1")
if args.lengthscale_lower_bound_min <= 0:
    parser.error("--lengthscale-lower-bound-min must be positive")
if args.lengthscale_lower_bound_max < args.lengthscale_lower_bound_min:
    parser.error(
        "--lengthscale-lower-bound-max must be greater than or equal to "
        "--lengthscale-lower-bound-min"
    )

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
best_trial_state = {"value": float("inf"), "state": None}


def train_with_lower_bounds(lower_bounds, trial):
    """Train one GP trial and return its best validation-loss checkpoint."""
    # Every trial starts from the same random state, making the bound the only
    # intended source of variation in the optimization objective.
    torch.manual_seed(42)

    X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
    y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)
    X_scaler.train()
    Xtrain_scaled = X_scaler(Xtrain)

    likelihood = GaussianLikelihood(batch_shape=batch_shape).to(device)
    with torch.no_grad():
        y_scaler.train()
        res = y_scaler((ytrain - mean(Xtrain)).unsqueeze(-1)).squeeze(-1)
    model = model_class(
        Xtrain_scaled,
        res,
        likelihood,
        batch_shape=batch_shape,
        lengthscale_lower_bounds=lower_bounds,
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

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Trial %d, epoch %d: train loss %.6f, validation loss %.6f, noise %.2e",
                trial.number,
                epoch + 1,
                train_loss.item(),
                current_val_loss,
                likelihood.noise.mean().item(),
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    trial.set_user_attr("best_epoch", epoch + 1 - epochs_without_improvement)
    return best_val_loss, best_state


def objective(trial):
    lower_bounds = tuple(
        trial.suggest_float(
            f"lengthscale_lower_bound_{dimension}",
            args.lengthscale_lower_bound_min,
            args.lengthscale_lower_bound_max,
            log=True,
        )
        for dimension in range(3)
    )
    val_loss, state = train_with_lower_bounds(lower_bounds, trial)
    if val_loss < best_trial_state["value"]:
        best_trial_state["value"] = val_loss
        best_trial_state["state"] = state
    return val_loss


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(
    direction="minimize",
    study_name=args.out.stem,
    storage=(
        f"sqlite:///{args.storage_name}.db" if args.storage_name is not None else None
    ),
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.n_trials)
best_trial = study.best_trial
best_state = best_trial_state["state"]
best_lower_bounds = tuple(
    best_trial.params[f"lengthscale_lower_bound_{dimension}"] for dimension in range(3)
)
logger.info(
    "Best lengthscale lower bounds: %s (validation loss: %.6f)",
    best_lower_bounds,
    best_trial.value,
)

X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=batch_shape).to(device)
y_scaler = scaler_module.StandardScaler(1, batch_shape=batch_shape).to(device)
X_scaler.train()
Xtrain_scaled = X_scaler(Xtrain)
with torch.no_grad():
    y_scaler.train()
    res = y_scaler((ytrain - mean(Xtrain)).unsqueeze(-1)).squeeze(-1)
likelihood = GaussianLikelihood(batch_shape=batch_shape).to(device)
model = model_class(
    Xtrain_scaled,
    res,
    likelihood,
    batch_shape=batch_shape,
    lengthscale_lower_bounds=best_lower_bounds,
).to(device)
X_scaler.load_state_dict(best_state["X_scaler"])
y_scaler.load_state_dict(best_state["y_scaler"])
likelihood.load_state_dict(best_state["likelihood"])
model.load_state_dict(best_state["model"])

save_gpr(
    Xtrain,
    ytrain,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    args.out,
)
