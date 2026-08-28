import torch
from gpytorch.mlls import VariationalELBO

from models.v1.feature_models import gpr as model_module
from models.v1.feature_models import load as load_module
from models.v1.feature_models import scale as scaler_module
from models.v1.feature_models.likelihoods import GaussianLikelihood
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
from scripts.v1.train.save import save_gpr

logger = configure_logging(__name__)
torch.manual_seed(42)

HYPERPARAMETER_DEFAULTS = PRIOR_HYPERPARAMETER_DEFAULTS.copy()

parser = create_train_parser(HYPERPARAMETER_DEFAULTS)
args = parser.parse_args()
optimized_hyperparameters = set(args.optimize_hyperparameters)
controls = validate_train_args(parser, args, logger)
has_validation = controls.has_validation
device = controls.device

model_class = getattr(model_module, args.model)
TARGET = tuple(model_class.output_names)


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

final_mean = load_module.load_PriorMean(path=args.prior_mean, device=device)
final_mean.eval()
Xtrain = expand_output_batch(Xtrain_base)  # (*K, 3, N, D)

if has_validation:
    Xval_base, yval = load_data(args.Xval, args.yval)
    if Xval_base.shape[:-2] != external_batch_shape:
        parser.error("Training and validation data must have the same batch shape.")
    Xval = expand_output_batch(Xval_base)

    # The saved prior mean is the final train-validation refit. Fit a separate
    # fold-batched prior mean on fold-train rows only for leakage-free CV.
    torch.manual_seed(42)
    cv_mean = final_mean.__class__(batch_shape=external_batch_shape).to(device)
    cv_mean.train()
    cv_mean_optimizer = torch.optim.Adam(cv_mean.parameters(), lr=args.learning_rate)
    cv_mean_loss = torch.nn.MSELoss()
    for _ in range(args.num_epochs):
        cv_mean_optimizer.zero_grad()
        loss = cv_mean_loss(cv_mean(Xtrain_base), ytrain)
        loss.backward()
        cv_mean_optimizer.step()
    cv_mean.eval()
else:
    cv_mean = final_mean

with torch.no_grad():
    res_train = ytrain - cv_mean(Xtrain_base)
    if has_validation:
        res_val = yval - cv_mean(Xval_base)

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
    trial=None,
):
    """Train all output batches under one shared HPO configuration."""
    torch.manual_seed(42)
    likelihood = GaussianLikelihood(
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
        batch_shape=gp_batch_shape,
    ).to(device)
    model = model_class(
        inducing_points=Xtrain_inducing_points.clone().detach(),
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
        validation_reduce_dims=-1,
        trial=trial,
    )


def objective(trial):
    hyperparameters = suggest_prior_hyperparameters(
        trial, HYPERPARAMETER_DEFAULTS, optimized_hyperparameters, args
    )
    duplicate_value = reuse_duplicate_trial(trial, logger)
    if duplicate_value is not None:
        return duplicate_value
    return train_with_hyperparameters(**hyperparameters, trial=trial)


def train_on_all_data(
    noise_prior_loc,
    noise_prior_scale,
    lengthscale_prior_loc,
    lengthscale_prior_scale,
    num_epochs,
    lr_reductions,
):
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

    likelihood = GaussianLikelihood(
        batch_shape=refit_batch_shape,
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
    ).to(device)

    X_scaler = scaler_module.MinMaxScaler(dim, batch_shape=refit_batch_shape).to(device)
    X_scaler.train()
    with torch.no_grad():
        Xall_scaled = X_scaler(Xall)
    X_scaler.eval()

    y_scaler = scaler_module.StandardScaler(1, batch_shape=refit_batch_shape).to(device)
    y_scaler.train()
    with torch.no_grad():
        res_all = yall - final_mean(Xall_base)
        res_all_scaled = y_scaler(res_all.unsqueeze(-1)).squeeze(-1)
    y_scaler.eval()

    model = model_class(
        inducing_points=unique_inducing_points_per_fold(Xall_scaled),
        batch_shape=refit_batch_shape,
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
        cholesky_jitter=args.cholesky_jitter,
        max_grad_norm=args.max_grad_norm,
        num_epochs=num_epochs,
        lr_reductions=lr_reductions,
        logger=logger,
    )
    return X_scaler, y_scaler, likelihood, model


def fit_cross_validation_model(
    noise_prior_loc,
    noise_prior_scale,
    lengthscale_prior_loc,
    lengthscale_prior_scale,
    num_epochs,
    lr_reductions,
):
    """Refit fold batches on fold-train data for leakage-free metadata."""
    torch.manual_seed(42)
    likelihood = GaussianLikelihood(
        batch_shape=gp_batch_shape,
        noise_prior_loc=noise_prior_loc,
        noise_prior_scale=noise_prior_scale,
    ).to(device)
    model = model_class(
        inducing_points=Xtrain_inducing_points.clone().detach(),
        batch_shape=gp_batch_shape,
        lengthscale_prior_loc=lengthscale_prior_loc,
        lengthscale_prior_scale=lengthscale_prior_scale,
    ).to(device)
    fit_all_data(
        likelihood=likelihood,
        model=model,
        mll=VariationalELBO(likelihood, model, num_data=num_data),
        parameters=list(model.parameters()) + list(likelihood.parameters()),
        X=Xtrain_scaled,
        y=res_train_scaled,
        learning_rate=args.learning_rate,
        cholesky_jitter=args.cholesky_jitter,
        max_grad_norm=args.max_grad_norm,
        num_epochs=num_epochs,
        lr_reductions=lr_reductions,
        logger=logger,
    )
    return model


study = optimize_or_load_study(
    args=args,
    controls=controls,
    objective=objective,
    optimized_hyperparameters=optimized_hyperparameters,
    baseline_params=prior_baseline(args),
)
best_trial, best_hyperparameters, best_epoch, best_lr_reductions = select_best_trial(
    study, HYPERPARAMETER_DEFAULTS, optimized_hyperparameters, controls, args
)
logger.info(
    "Best shared hyperparameters: %s (validation loss: %.6f)",
    best_hyperparameters,
    best_trial.value,
)

metadata = {}
if has_validation:
    cv_model = fit_cross_validation_model(
        best_hyperparameters["noise_prior_loc"],
        best_hyperparameters["noise_prior_scale"],
        best_hyperparameters["lengthscale_prior_loc"],
        best_hyperparameters["lengthscale_prior_scale"],
        best_epoch,
        best_lr_reductions,
    )
    cv_lengthscale = cv_model.covar_module.base_kernel.lengthscale.detach().clone()
    metadata["cross_validation"] = {
        "lengthscale": cv_lengthscale,
        "prior_mean": {
            "type": cv_mean.__class__.__name__,
            "args": {"batch_shape": cv_mean.batch_shape},
            "state_dict": {
                name: value.detach().clone()
                for name, value in cv_mean.state_dict().items()
            },
        },
        "X_scaler": {
            "type": X_scaler.__class__.__name__,
            "args": {
                "dim": X_scaler.dim,
                "batch_shape": X_scaler.batch_shape,
            },
            "state_dict": {
                name: value.detach().clone()
                for name, value in X_scaler.state_dict().items()
            },
        },
        "y_scaler": {
            "type": y_scaler.__class__.__name__,
            "args": {
                "dim": y_scaler.dim,
                "batch_shape": y_scaler.batch_shape,
            },
            "state_dict": {
                name: value.detach().clone()
                for name, value in y_scaler.state_dict().items()
            },
        },
        "model": {
            "type": cv_model.__class__.__name__,
            "args": {
                "inducing_points": cv_model.inducing_points.detach().clone(),
                "lengthscale_prior_loc": cv_model.lengthscale_prior_loc,
                "lengthscale_prior_scale": cv_model.lengthscale_prior_scale,
                "batch_shape": cv_model.batch_shape,
            },
            "state_dict": {
                name: value.detach().clone()
                for name, value in cv_model.state_dict().items()
            },
        },
    }
    logger.info(
        "Saved fold-specific GPR lengthscales with shape %s.",
        tuple(cv_lengthscale.shape),
    )

X_scaler, y_scaler, likelihood, model = train_on_all_data(
    best_hyperparameters["noise_prior_loc"],
    best_hyperparameters["noise_prior_scale"],
    best_hyperparameters["lengthscale_prior_loc"],
    best_hyperparameters["lengthscale_prior_scale"],
    best_epoch,
    best_lr_reductions,
)
save_gpr(X_scaler, y_scaler, likelihood, model, args.out, metadata=metadata)
