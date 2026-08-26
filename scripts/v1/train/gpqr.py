import numpy as np
import torch
from gpytorch.mlls import VariationalELBO

from models.v1.feature_models import gpqr as model_module
from models.v1.feature_models import load as load_module
from models.v1.feature_models import scale as scaler_module
from models.v1.feature_models.likelihoods import CenterGapQuantilesLikelihood
from scripts.v1.train.batch import load_batched_arrays
from scripts.v1.train.common import (
    PRIOR_HYPERPARAMETER_DEFAULTS,
    configure_logging,
    create_train_parser,
    fit_all_data,
    fit_trial,
    optimize_or_load_study,
    prior_baseline,
    reuse_duplicate_trial,
    select_best_trial,
    suggest_prior_hyperparameters,
    validate_train_args,
)
from scripts.v1.train.inducing import unique_inducing_points_per_fold
from scripts.v1.train.save import save_gpqr

logger = configure_logging(__name__)
torch.manual_seed(42)

HYPERPARAMETER_DEFAULTS = PRIOR_HYPERPARAMETER_DEFAULTS | {"num_latents": None}

parser = create_train_parser(HYPERPARAMETER_DEFAULTS, quantiles=True)
args = parser.parse_args()
optimized_hyperparameters = set(args.optimize_hyperparameters)
controls = validate_train_args(parser, args, logger)
has_validation = controls.has_validation
device = controls.device

model_class = getattr(model_module, args.model)
TARGET = tuple(model_class.output_names)
quantiles = torch.tensor(args.quantiles, dtype=torch.float32, device=device)
central_quantile_idx = int(np.argmin(np.abs(quantiles.cpu().numpy() - 0.5)))
num_quantiles = len(quantiles)
num_lower_quantiles = central_quantile_idx
HYPERPARAMETER_DEFAULTS["num_latents"] = num_quantiles


def load_data(X_path, y_path):
    try:
        X_arr, y_arr = load_batched_arrays(
            X_path, y_path, TARGET, args.index_col, args.batch_col
        )
    except ValueError as exc:
        parser.error(str(exc))
    return (
        torch.tensor(X_arr, dtype=torch.float32, device=device),
        torch.tensor(y_arr, dtype=torch.float32, device=device),
    )


def expand_output_batch(X):
    return X.unsqueeze(-3).expand(*X.shape[:-2], len(TARGET), *X.shape[-2:])


Xtrain_base, ytrain = load_data(args.Xtrain, args.ytrain)
external_batch_shape = Xtrain_base.shape[:-2]
gp_batch_shape = torch.Size((*external_batch_shape, len(TARGET)))

mean = load_module.load_PriorMean(path=args.prior_mean, device=device)
mean.eval()
with torch.no_grad():
    res_train = ytrain - mean(Xtrain_base)
Xtrain = expand_output_batch(Xtrain_base)

if has_validation:
    Xval_base, yval = load_data(args.Xval, args.yval)
    if Xval_base.shape[:-2] != external_batch_shape:
        parser.error("Training and validation data must have the same batch shape.")
    with torch.no_grad():
        res_val = yval - mean(Xval_base)
    Xval = expand_output_batch(Xval_base)

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=gp_batch_shape).to(device)
X_scaler.train()
with torch.no_grad():
    Xtrain_scaled = X_scaler(Xtrain)
X_scaler.eval()

y_scaler = scaler_module.StandardScaler(1, batch_shape=gp_batch_shape).to(device)
y_scaler.train()
with torch.no_grad():
    res_train_scaled = y_scaler(res_train.unsqueeze(-1)).squeeze(-1)
y_scaler.eval()

if has_validation:
    with torch.no_grad():
        Xval_scaled = X_scaler(Xval)
        res_val_scaled = y_scaler(res_val.unsqueeze(-1)).squeeze(-1)

Xtrain_inducing_points = unique_inducing_points_per_fold(Xtrain_scaled)


def train_with_hyperparameters(
    noise_prior_loc,
    noise_prior_scale,
    lengthscale_prior_loc,
    lengthscale_prior_scale,
    num_latents,
    trial=None,
):
    torch.manual_seed(42)
    likelihood = CenterGapQuantilesLikelihood(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=gp_batch_shape,
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
    ).to(device)
    model = model_class(
        inducing_points=Xtrain_inducing_points.clone().detach(),
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_latents,
        batch_shape=gp_batch_shape,
        lengthscale_prior_loc=lengthscale_prior_loc,
        lengthscale_prior_scale=lengthscale_prior_scale,
    ).to(device)

    return fit_trial(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=num_data),
        parameters=list(model.parameters()) + list(likelihood.parameters()),
        Xtrain=Xtrain_scaled,
        ytrain=res_train_scaled,
        Xval=Xval_scaled,
        yval=res_val_scaled,
        args=args,
        controls=controls,
        logger=logger,
        validation_reduce_dims=(-2, -1),
        trial=trial,
        num_likelihood_samples=args.num_likelihood_samples,
    )


def normalized_hyperparameters(trial=None):
    hyperparameters = (
        suggest_prior_hyperparameters(
            trial, HYPERPARAMETER_DEFAULTS, optimized_hyperparameters, args
        )
        if trial is not None
        else HYPERPARAMETER_DEFAULTS.copy()
    )
    hyperparameters["num_latents"] = int(
        np.clip(hyperparameters["num_latents"], 2, num_quantiles)
    )
    if trial is not None and "num_latents" in optimized_hyperparameters:
        hyperparameters["num_latents"] = trial.suggest_int(
            "num_latents", 2, num_quantiles
        )
    return hyperparameters


def objective(trial):
    hyperparameters = normalized_hyperparameters(trial)
    duplicate_value = reuse_duplicate_trial(trial, logger)
    if duplicate_value is not None:
        return duplicate_value
    return train_with_hyperparameters(**hyperparameters, trial=trial)


def train_on_all_data(hyperparameters, num_epochs, lr_reductions):
    torch.manual_seed(42)
    if has_validation:
        Xall_base = torch.cat((Xtrain_base, Xval_base), dim=-2)
        yall = torch.cat((ytrain, yval), dim=-1)
    else:
        Xall_base = Xtrain_base
        yall = ytrain
    Xall = expand_output_batch(Xall_base)

    likelihood = CenterGapQuantilesLikelihood(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=gp_batch_shape,
        noise_prior_loc=hyperparameters["noise_prior_loc"],
        noise_prior_scale=hyperparameters["noise_prior_scale"],
    ).to(device)

    X_scaler.train()
    with torch.no_grad():
        Xall_scaled = X_scaler(Xall)
    X_scaler.eval()

    y_scaler.train()
    with torch.no_grad():
        res_all = yall - mean(Xall_base)
        res_all_scaled = y_scaler(res_all.unsqueeze(-1)).squeeze(-1)
    y_scaler.eval()

    model = model_class(
        inducing_points=unique_inducing_points_per_fold(Xall_scaled),
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=hyperparameters["num_latents"],
        batch_shape=gp_batch_shape,
        lengthscale_prior_loc=hyperparameters["lengthscale_prior_loc"],
        lengthscale_prior_scale=hyperparameters["lengthscale_prior_scale"],
    ).to(device)

    fit_all_data(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=Xall.shape[-2]),
        parameters=list(model.parameters()) + list(likelihood.parameters()),
        X=Xall_scaled,
        y=res_all_scaled,
        learning_rate=args.learning_rate,
        cholesky_jitter=args.cholesky_jitter,
        max_grad_norm=args.max_grad_norm,
        num_epochs=num_epochs,
        lr_reductions=lr_reductions,
        logger=logger,
        num_likelihood_samples=args.num_likelihood_samples,
    )
    return X_scaler, y_scaler, likelihood, model


default_hyperparameters = normalized_hyperparameters()
baseline_params = prior_baseline(args)
baseline_params["num_latents"] = default_hyperparameters["num_latents"]
study = optimize_or_load_study(
    args=args,
    controls=controls,
    objective=objective,
    optimized_hyperparameters=optimized_hyperparameters,
    baseline_params=baseline_params,
)
best_trial, best_hyperparameters, best_epoch, best_lr_reductions = select_best_trial(
    study, default_hyperparameters, optimized_hyperparameters, controls, args
)
logger.info(
    "Best shared hyperparameters: %s (validation loss: %.6f)",
    best_hyperparameters,
    best_trial.value,
)

X_scaler, y_scaler, likelihood, model = train_on_all_data(
    best_hyperparameters, best_epoch, best_lr_reductions
)
save_gpqr(quantiles, X_scaler, y_scaler, likelihood, model, args.out)
