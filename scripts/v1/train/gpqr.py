import numpy as np
import torch
from gpytorch.mlls import VariationalELBO
from gpytorch_qr.settings import quantile_gap_lower_bound

from models.v1.feature_models import gpqr as model_module
from models.v1.feature_models import gpr as gpr_module
from models.v1.feature_models import load as load_module
from models.v1.feature_models import prior as prior_module
from models.v1.feature_models import scale as scaler_module
from models.v1.feature_models.likelihoods import CenterGapQuantilesLikelihood
from scripts.v1.train.batch import load_batched_arrays
from scripts.v1.train.common import (
    configure_logging,
    create_train_parser,
    fit_all_data,
    fit_with_validation,
    validate_train_args,
)
from scripts.v1.train.inducing import unique_inducing_points_per_fold
from scripts.v1.train.save import save_gpqr

logger = configure_logging(__name__)
torch.manual_seed(42)

parser = create_train_parser({}, quantiles=True)
parser.add_argument(
    "--gpr-model",
    type=str,
    required=True,
    help=(
        "Fitted GPR checkpoint whose input scaler and ARD lengthscale are "
        "reused and fixed in the GPQR."
    ),
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
    "--quantile-gap-lower-bound",
    type=float,
    default=1e-4,
    help="Lower bound on (f(q2) - f(q1)) / (q2 - q1) in scaled response units.",
)
args = parser.parse_args()
controls = validate_train_args(parser, args, logger)
has_validation = controls.has_validation
device = controls.device
if args.num_epochs is None:
    parser.error("--num-epochs is required for GPQR training")
if args.batch_size is not None and args.batch_size <= 0:
    parser.error("--batch-size must be greater than zero")
try:
    quantile_gap_context = quantile_gap_lower_bound(args.quantile_gap_lower_bound)
except ValueError as exc:
    parser.error(str(exc))

model_class = getattr(model_module, args.model)
TARGET = tuple(model_class.output_names)
quantiles = torch.tensor(args.quantiles, dtype=torch.float32, device=device)
central_quantile_idx = int(np.argmin(np.abs(quantiles.cpu().numpy() - 0.5)))
num_quantiles = len(quantiles)


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
    _,
    _,
    final_gpr_model,
    gpr_metadata,
) = load_module.load_GPR(path=args.gpr_model, device=device, return_metadata=True)
final_X_scaler.eval()
final_gpr_model.eval()
final_gpr_batch_shape = torch.Size((len(TARGET),))
if (
    final_X_scaler.batch_shape != final_gpr_batch_shape
    or final_gpr_model.batch_shape != final_gpr_batch_shape
):
    parser.error("the fitted GPR model and X scaler must have one batch per output")
final_lengthscale = (
    final_gpr_model.covar_module.base_kernel.lengthscale.detach().clone()
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
expected_lengthscale_shape = (len(TARGET), 1, dim)
if tuple(final_lengthscale.shape) != expected_lengthscale_shape:
    parser.error(
        "the fitted GPR must contain one ARD lengthscale vector per output; "
        f"expected {expected_lengthscale_shape}, got {tuple(final_lengthscale.shape)}"
    )

if has_validation:
    try:
        scaler_checkpoint = cv_metadata["X_scaler"]
        scaler_class = getattr(scaler_module, scaler_checkpoint["type"])
        cv_X_scaler = scaler_class(**scaler_checkpoint["args"]).to(device)
        cv_X_scaler.load_state_dict(scaler_checkpoint["state_dict"])

        model_checkpoint = cv_metadata["model"]
        cv_gpr_model_class = getattr(gpr_module, model_checkpoint["type"])
        cv_gpr_model = cv_gpr_model_class(**model_checkpoint["args"]).to(device)
        cv_gpr_model.load_state_dict(model_checkpoint["state_dict"])
        cv_lengthscale = (
            cv_gpr_model.covar_module.base_kernel.lengthscale.detach().clone()
        )
    except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
        parser.error(
            f"the GPR checkpoint has no valid cross-validation metadata: {exc}"
        )
    cv_X_scaler.eval()
    cv_gpr_model.eval()
    if (
        cv_X_scaler.batch_shape != gp_batch_shape
        or cv_gpr_model.batch_shape != gp_batch_shape
    ):
        parser.error("GPR metadata and GPQR data have different fold batch shapes")
    expected_cv_lengthscale_shape = (*gp_batch_shape, 1, dim)
    if tuple(cv_lengthscale.shape) != expected_cv_lengthscale_shape:
        parser.error(
            "the GPR cross-validation metadata has an invalid lengthscale shape: "
            f"expected {expected_cv_lengthscale_shape}, "
            f"got {tuple(cv_lengthscale.shape)}"
        )
else:
    cv_X_scaler = final_X_scaler
    cv_lengthscale = final_lengthscale


with torch.no_grad():
    res_train = ytrain - cv_mean(Xtrain_base)
    Xtrain_scaled = cv_X_scaler(Xtrain)

y_scaler = scaler_module.StandardScaler(1, batch_shape=gp_batch_shape).to(device)
y_scaler.train()
with torch.no_grad():
    res_train_scaled = y_scaler(res_train.unsqueeze(-1)).squeeze(-1)
y_scaler.eval()

if has_validation:
    with torch.no_grad():
        res_val = yval - cv_mean(Xval_base)
        Xval_scaled = cv_X_scaler(Xval)
        res_val_scaled = y_scaler(res_val.unsqueeze(-1)).squeeze(-1)

Xtrain_inducing_points = unique_inducing_points_per_fold(Xtrain_scaled)


def train_on_all_data(num_epochs, lr_reductions):
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
    ).to(device)

    with torch.no_grad():
        Xall_scaled = final_X_scaler(Xall)

    y_scaler = scaler_module.StandardScaler(1, batch_shape=refit_batch_shape).to(device)
    y_scaler.train()
    with torch.no_grad():
        res_all = yall - final_mean(Xall_base)
        res_all_scaled = y_scaler(res_all.unsqueeze(-1)).squeeze(-1)
    y_scaler.eval()

    model = model_class(
        inducing_points=unique_inducing_points_per_fold(Xall_scaled),
        num_quantiles=num_quantiles,
        central_quantile_idx=central_quantile_idx,
        num_latents=num_quantiles,
        batch_shape=refit_batch_shape,
        lengthscale=final_lengthscale,
        quantile_levels=quantiles,
    ).to(device)

    parameters = [
        parameter
        for parameter in (*model.parameters(), *likelihood.parameters())
        if parameter.requires_grad
    ]

    fit_all_data(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=Xall.shape[-2]),
        parameters=parameters,
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


def select_training_schedule():
    torch.manual_seed(42)
    likelihood = CenterGapQuantilesLikelihood(
        quantile_levels=quantiles,
        central_quantile_idx=central_quantile_idx,
        batch_shape=gp_batch_shape,
    ).to(device)
    model = model_class(
        inducing_points=Xtrain_inducing_points.clone().detach(),
        num_quantiles=num_quantiles,
        central_quantile_idx=central_quantile_idx,
        num_latents=num_quantiles,
        batch_shape=gp_batch_shape,
        lengthscale=cv_lengthscale,
        quantile_levels=quantiles,
    ).to(device)
    parameters = [
        parameter
        for parameter in (*model.parameters(), *likelihood.parameters())
        if parameter.requires_grad
    ]
    _, best_epoch, lr_reductions = fit_with_validation(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=num_data),
        parameters=parameters,
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
    return best_epoch, lr_reductions


logger.info(
    "Fixing final GPQR lengthscale to fitted GPR ARD values: %s",
    final_lengthscale.squeeze(-2).cpu().tolist(),
)
if has_validation:
    logger.info(
        "Using fold-specific GPR lengthscales for GPQR early stopping: shape %s",
        tuple(cv_lengthscale.shape),
    )
if args.batch_size is not None:
    logger.info(
        "Using GPQR observation minibatches of at most %d rows.", args.batch_size
    )

with quantile_gap_context:
    if has_validation:
        best_epoch, best_lr_reductions = select_training_schedule()
        logger.info("Selected GPQR training length: %d epochs", best_epoch)
    else:
        best_epoch, best_lr_reductions = args.num_epochs, []

    X_scaler, y_scaler, likelihood, model = train_on_all_data(
        best_epoch, best_lr_reductions
    )
    save_gpqr(
        quantiles,
        X_scaler,
        y_scaler,
        likelihood,
        model,
        args.out,
        quantile_gap_lower_bound=args.quantile_gap_lower_bound,
    )
