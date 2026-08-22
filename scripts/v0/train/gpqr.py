import numpy as np
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
from scripts.v0.train.common import (
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
from scripts.v0.train.inducing import unique_inducing_points_per_fold
from scripts.v0.train.save import save_gpqr

logger = configure_logging(__name__)

torch.manual_seed(42)

HYPERPARAMETER_DEFAULTS = PRIOR_HYPERPARAMETER_DEFAULTS | {"num_latents": None}


parser = create_train_parser(HYPERPARAMETER_DEFAULTS, target=True, quantiles=True)
args = parser.parse_args()
optimized_hyperparameters = set(args.optimize_hyperparameters)
controls = validate_train_args(parser, args, logger)
has_validation = controls.has_validation
device = controls.device

quantiles = torch.tensor(args.quantiles, dtype=torch.float32, device=device)
central_quantile_idx = np.argmin(np.abs(quantiles.detach().cpu().numpy() - 0.5))
num_quantiles = len(quantiles)
num_lower_quantiles = central_quantile_idx

HYPERPARAMETER_DEFAULTS["num_latents"] = num_quantiles

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


def objective(trial):
    hyperparameters = suggest_prior_hyperparameters(
        trial, HYPERPARAMETER_DEFAULTS, optimized_hyperparameters, args
    )
    hyperparameters["num_latents"] = int(
        np.clip(hyperparameters["num_latents"], 2, num_quantiles)
    )
    if "num_latents" in optimized_hyperparameters:
        hyperparameters["num_latents"] = trial.suggest_int(
            "num_latents", 2, num_quantiles
        )
    duplicate_value = reuse_duplicate_trial(trial, logger)
    if duplicate_value is not None:
        return duplicate_value
    val_loss = train_with_hyperparameters(
        **hyperparameters,
        trial=trial,
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

    fit_all_data(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=Xall.shape[-2]),
        parameters=list(model.parameters()) + list(likelihood.parameters()),
        X=Xall_scaled,
        y=res_all_scaled,
        learning_rate=args.learning_rate,
        num_epochs=num_epochs,
        lr_reductions=lr_reductions,
        logger=logger,
        num_likelihood_samples=args.num_likelihood_samples,
    )

    return X_scaler, y_scaler, likelihood, model


default_hyperparameters = HYPERPARAMETER_DEFAULTS.copy()
default_hyperparameters["num_latents"] = int(
    np.clip(default_hyperparameters["num_latents"], 2, num_quantiles)
)
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
best_noise_prior_loc = best_hyperparameters["noise_prior_loc"]
best_noise_prior_scale = best_hyperparameters["noise_prior_scale"]
best_lengthscale_prior_loc = best_hyperparameters["lengthscale_prior_loc"]
best_lengthscale_prior_scale = best_hyperparameters["lengthscale_prior_scale"]
best_num_latents = best_hyperparameters["num_latents"]
logger.info(
    "Best hyperparameters: noise_prior_loc=%s, noise_prior_scale=%s, "
    "lengthscale_prior_loc=%s, lengthscale_prior_scale=%s, num_latents=%d "
    "(validation loss: %.6f)",
    best_noise_prior_loc,
    best_noise_prior_scale,
    best_lengthscale_prior_loc,
    best_lengthscale_prior_scale,
    best_num_latents,
    best_trial.value,
)

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
