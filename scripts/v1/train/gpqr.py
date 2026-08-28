import numpy as np
import optuna
import torch
from gpytorch.mlls import VariationalELBO

from models.v1.feature_models import gpqr as model_module
from models.v1.feature_models import gpr as gpr_module
from models.v1.feature_models import load as load_module
from models.v1.feature_models import prior as prior_module
from models.v1.feature_models import scale as scaler_module
from models.v1.feature_models.likelihoods import CenterGapQuantilesLikelihood
from models.v1.feature_models.quantile import DEFAULT_QUANTILE_SLOPE_LOWER_BOUND
from scripts.v1.train.batch import load_batched_arrays
from scripts.v1.train.common import (
    PRIOR_HYPERPARAMETER_DEFAULTS,
    configure_logging,
    create_train_parser,
    fit_all_data,
    fit_trial,
    initialize_optuna_storage,
    validate_train_args,
)
from scripts.v1.train.inducing import unique_inducing_points_per_fold
from scripts.v1.train.save import save_gpqr

logger = configure_logging(__name__)
torch.manual_seed(42)

HYPERPARAMETER_DEFAULTS = PRIOR_HYPERPARAMETER_DEFAULTS.copy()

parser = create_train_parser(HYPERPARAMETER_DEFAULTS, quantiles=True)
parser.add_argument(
    "--gpr-model",
    type=str,
    required=True,
    help="Fitted GPR checkpoint used as the GPQR mean function.",
)
parser.add_argument(
    "--gpr-storage",
    type=str,
    required=True,
    help="Optuna storage containing the GPR hyperparameter study.",
)
parser.add_argument(
    "--gpr-study-name",
    type=str,
    required=True,
    help="Name of the GPR Optuna study to reuse.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help=(
        "Observation minibatch size for GPQR optimizer steps and validation. "
        "Scalers and inducing points still use the complete dataset."
    ),
)
parser.add_argument(
    "--quantile-slope-lower-bound",
    type=float,
    default=DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    help="Lower bound on (f(q2) - f(q1)) / (q2 - q1) in scaled response units.",
)
args = parser.parse_args()
controls = validate_train_args(parser, args, logger)
has_validation = controls.has_validation
device = controls.device
if args.optimize_hyperparameters:
    parser.error("GPQR does not optimize hyperparameters; use the GPR study")
if args.num_epochs is None:
    parser.error("--num-epochs is required for GPQR final training")
if args.batch_size is not None and args.batch_size <= 0:
    parser.error("--batch-size must be greater than zero")
if (
    not np.isfinite(args.quantile_slope_lower_bound)
    or args.quantile_slope_lower_bound <= 0
):
    parser.error("--quantile-slope-lower-bound must be finite and positive")

model_class = getattr(model_module, args.model)
TARGET = tuple(model_class.output_names)
quantiles = torch.tensor(args.quantiles, dtype=torch.float32, device=device)
central_quantile_idx = int(np.argmin(np.abs(quantiles.cpu().numpy() - 0.5)))
num_quantiles = len(quantiles)
num_lower_quantiles = central_quantile_idx


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

(
    final_X_scaler,
    final_gpr_y_scaler,
    _,
    final_gpr_model,
    gpr_metadata,
) = load_module.load_GPR(path=args.gpr_model, device=device, return_metadata=True)
final_X_scaler.eval()
final_gpr_y_scaler.eval()
final_gpr_model.eval()
final_gpr_batch_shape = torch.Size((len(TARGET),))
if (
    final_X_scaler.batch_shape != final_gpr_batch_shape
    or final_gpr_y_scaler.batch_shape != final_gpr_batch_shape
    or final_gpr_model.batch_shape != final_gpr_batch_shape
):
    parser.error(
        "the fitted GPR model and scalers must have one batch per output variable"
    )

final_mean = load_module.load_PriorMean(path=args.prior_mean, device=device)
final_mean.eval()
if has_validation:
    try:
        cv_metadata = gpr_metadata["cross_validation"]
        mean_checkpoint = cv_metadata["prior_mean"]
        mean_class = getattr(prior_module, mean_checkpoint["type"])
        cv_mean = mean_class(**mean_checkpoint["args"]).to(device)
        cv_mean.load_state_dict(mean_checkpoint["state_dict"])
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        parser.error(
            f"the GPR checkpoint has no valid cross-validation prior mean: {exc}"
        )
    cv_mean.eval()
    if cv_mean.batch_shape != external_batch_shape:
        parser.error(
            "GPR metadata and GPQR data have different prior-mean batch shapes: "
            f"{tuple(cv_mean.batch_shape)} != {tuple(external_batch_shape)}"
        )
else:
    cv_mean = final_mean

Xtrain = expand_output_batch(Xtrain_base)

if has_validation:
    Xval_base, yval = load_data(args.Xval, args.yval)
    if Xval_base.shape[:-2] != external_batch_shape:
        parser.error("Training and validation data must have the same batch shape.")
    Xval = expand_output_batch(Xval_base)

dim = Xtrain.shape[-1]
num_data = Xtrain.shape[-2]
if final_X_scaler.dim != dim:
    parser.error(
        f"the fitted GPR expects {final_X_scaler.dim} features, but the data has {dim}"
    )

if has_validation:
    try:
        scaler_checkpoint = cv_metadata["X_scaler"]
        scaler_class = getattr(scaler_module, scaler_checkpoint["type"])
        cv_X_scaler = scaler_class(**scaler_checkpoint["args"]).to(device)
        cv_X_scaler.load_state_dict(scaler_checkpoint["state_dict"])

        y_scaler_checkpoint = cv_metadata["y_scaler"]
        y_scaler_class = getattr(scaler_module, y_scaler_checkpoint["type"])
        cv_gpr_y_scaler = y_scaler_class(**y_scaler_checkpoint["args"]).to(device)
        cv_gpr_y_scaler.load_state_dict(y_scaler_checkpoint["state_dict"])

        model_checkpoint = cv_metadata["model"]
        cv_gpr_model_class = getattr(gpr_module, model_checkpoint["type"])
        cv_gpr_model = cv_gpr_model_class(**model_checkpoint["args"]).to(device)
        cv_gpr_model.load_state_dict(model_checkpoint["state_dict"])
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        parser.error(
            f"the GPR checkpoint has no valid cross-validation metadata: {exc}"
        )
    cv_X_scaler.eval()
    cv_gpr_y_scaler.eval()
    cv_gpr_model.eval()
    if (
        cv_X_scaler.batch_shape != gp_batch_shape
        or cv_gpr_y_scaler.batch_shape != gp_batch_shape
        or cv_gpr_model.batch_shape != gp_batch_shape
    ):
        parser.error("GPR metadata and GPQR data have different fold batch shapes")
else:
    cv_X_scaler = final_X_scaler
    cv_gpr_y_scaler = final_gpr_y_scaler
    cv_gpr_model = final_gpr_model


def gpr_posterior_mean(X_base, prior_mean, X_scaler, y_scaler, model):
    scaled_posterior = model(X_scaler(expand_output_batch(X_base)))
    return gpr_module.posterior_mean(
        prior_mean(X_base),
        scaled_posterior,
        y_scaler,
    )


with torch.no_grad():
    res_train = ytrain - gpr_posterior_mean(
        Xtrain_base, cv_mean, cv_X_scaler, cv_gpr_y_scaler, cv_gpr_model
    )
    Xtrain_scaled = cv_X_scaler(Xtrain)

y_scaler = scaler_module.StandardScaler(1, batch_shape=gp_batch_shape).to(device)
y_scaler.train()
with torch.no_grad():
    res_train_scaled = y_scaler(res_train.unsqueeze(-1)).squeeze(-1)
y_scaler.eval()

if has_validation:
    with torch.no_grad():
        res_val = yval - gpr_posterior_mean(
            Xval_base, cv_mean, cv_X_scaler, cv_gpr_y_scaler, cv_gpr_model
        )
        Xval_scaled = cv_X_scaler(Xval)
        res_val_scaled = y_scaler(res_val.unsqueeze(-1)).squeeze(-1)

Xtrain_inducing_points = unique_inducing_points_per_fold(Xtrain_scaled)


def train_on_all_data(hyperparameters, num_epochs, lr_reductions):
    torch.manual_seed(42)
    if has_validation:
        # Every fold contains the same train-validation union. Refit one
        # unbatched model from the first fold instead of retaining K redundant
        # fold-specific models in the saved checkpoint.
        Xall_base = torch.cat((Xtrain_base[0], Xval_base[0]), dim=-2)
        yall = torch.cat((ytrain[0], yval[0]), dim=-1)
        refit_batch_shape = torch.Size((len(TARGET),))
    else:
        Xall_base = Xtrain_base
        yall = ytrain
        refit_batch_shape = gp_batch_shape
    Xall = expand_output_batch(Xall_base)

    if final_mean.batch_shape != Xall_base.shape[:-2]:
        parser.error("refit prior-mean batch shape must match the refit training data")

    likelihood = CenterGapQuantilesLikelihood(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=refit_batch_shape,
        noise_prior_loc=hyperparameters["noise_prior_loc"],
        noise_prior_scale=hyperparameters["noise_prior_scale"],
        quantile_slope_lower_bound=args.quantile_slope_lower_bound,
    ).to(device)

    with torch.no_grad():
        Xall_scaled = final_X_scaler(Xall)

    y_scaler = scaler_module.StandardScaler(1, batch_shape=refit_batch_shape).to(device)
    y_scaler.train()
    with torch.no_grad():
        res_all = yall - gpr_posterior_mean(
            Xall_base,
            final_mean,
            final_X_scaler,
            final_gpr_y_scaler,
            final_gpr_model,
        )
        res_all_scaled = y_scaler(res_all.unsqueeze(-1)).squeeze(-1)
    y_scaler.eval()

    model = model_class(
        inducing_points=unique_inducing_points_per_fold(Xall_scaled),
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_quantiles,
        batch_shape=refit_batch_shape,
        lengthscale_prior_loc=hyperparameters["lengthscale_prior_loc"],
        lengthscale_prior_scale=hyperparameters["lengthscale_prior_scale"],
        quantile_levels=quantiles,
        quantile_slope_lower_bound=args.quantile_slope_lower_bound,
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
        batch_size=args.batch_size,
    )
    return final_X_scaler, y_scaler, likelihood, model


def select_final_schedule(hyperparameters, gpr_trial):
    if not has_validation:
        if "best_epoch" not in gpr_trial.user_attrs:
            parser.error("the GPR best trial has no early-stopping epoch")
        return (
            min(args.num_epochs, max(1, int(gpr_trial.user_attrs["best_epoch"]))),
            gpr_trial.user_attrs.get("lr_reductions", []),
        )

    # This is one validation run with GPR-derived prior hyperparameters, not
    # an Optuna trial. Its early-stopping schedule is reused for the final fit.
    torch.manual_seed(42)
    likelihood = CenterGapQuantilesLikelihood(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=gp_batch_shape,
        noise_prior_loc=hyperparameters["noise_prior_loc"],
        noise_prior_scale=hyperparameters["noise_prior_scale"],
        quantile_slope_lower_bound=args.quantile_slope_lower_bound,
    ).to(device)
    model = model_class(
        inducing_points=Xtrain_inducing_points.clone().detach(),
        num_quantiles=num_quantiles,
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_quantiles,
        batch_shape=gp_batch_shape,
        lengthscale_prior_loc=hyperparameters["lengthscale_prior_loc"],
        lengthscale_prior_scale=hyperparameters["lengthscale_prior_scale"],
        quantile_levels=quantiles,
        quantile_slope_lower_bound=args.quantile_slope_lower_bound,
    ).to(device)
    validation_loss, best_epoch, lr_reductions = fit_trial(
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
        num_likelihood_samples=args.num_likelihood_samples,
        batch_size=args.batch_size,
        return_training_details=True,
    )
    logger.info(
        "GPR-hyperparameter validation loss: %.6f (best epoch: %d)",
        validation_loss,
        best_epoch,
    )
    return best_epoch, lr_reductions


try:
    gpr_study = optuna.load_study(
        study_name=args.gpr_study_name,
        storage=initialize_optuna_storage(args.gpr_storage),
    )
except (KeyError, ValueError, optuna.exceptions.OptunaError) as exc:
    parser.error(f"cannot load GPR study: {exc}")

if args.batch_size is not None:
    logger.info(
        "Using GPQR observation minibatches of at most %d rows.", args.batch_size
    )

gpr_trial = gpr_study.best_trial
best_hyperparameters = {
    name: gpr_trial.params.get(name, default)
    for name, default in HYPERPARAMETER_DEFAULTS.items()
}
logger.info(
    "Reusing GPR hyperparameters: %s (validation loss: %.6f)",
    best_hyperparameters,
    gpr_trial.value,
)

X_scaler, y_scaler, likelihood, model = train_on_all_data(
    best_hyperparameters, *select_final_schedule(best_hyperparameters, gpr_trial)
)
save_gpqr(quantiles, X_scaler, y_scaler, likelihood, model, args.out)
