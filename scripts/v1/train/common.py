import argparse
import fcntl
import logging
import pathlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass

import gpytorch
import numpy as np
import optuna
import torch
from gpytorch.distributions import MultivariateNormal
from linear_operator import to_linear_operator
from linear_operator.operators import (
    BlockInterleavedLinearOperator,
    DiagLinearOperator,
)
from linear_operator.utils.errors import NotPSDError
from sqlalchemy.engine import make_url

MAX_DEFAULT_LR_SCHEDULER_PATIENCE = 50
DEFAULT_CHOLESKY_JITTER = 1e-4
DEFAULT_MAX_GRAD_NORM = 10.0
PRIOR_HYPERPARAMETER_DEFAULTS = {
    "noise_prior_loc": None,
    "noise_prior_scale": None,
    "lengthscale_prior_loc": None,
    "lengthscale_prior_scale": None,
}
PRIOR_HYPERPARAMETER_BASELINES = {
    "noise_prior_loc": -4.0,
    "noise_prior_scale": 0.5,
    "lengthscale_prior_loc": -1.0,
    "lengthscale_prior_scale": 0.5,
}


@dataclass(frozen=True)
class TrainingControls:
    has_validation: bool
    device: torch.device
    early_stopping_patience: int
    lr_scheduler_patience: int
    pruning_warmup: int
    pruning_patience: int
    pruning_interval: int


def configure_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(name)


def create_train_parser(hyperparameter_names, *, target=False, quantiles=False):
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group("input data and model")
    training_group = parser.add_argument_group("per-trial training")
    hpo_group = parser.add_argument_group("hyperparameter optimization")

    model_group.add_argument(
        "Xtrain", type=pathlib.Path, help="Training feature csv file."
    )
    model_group.add_argument(
        "ytrain", type=pathlib.Path, help="Training target csv file."
    )
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
    if target:
        model_group.add_argument("--target", type=str, help="Target variable name.")
    model_group.add_argument("--model", type=str, help="Model name.")
    if quantiles:
        model_group.add_argument(
            "--quantiles",
            type=float,
            nargs="+",
            help="Quantiles for the model.",
        )
    model_group.add_argument(
        "-o", "--out", type=pathlib.Path, help="Output model file."
    )
    model_group.add_argument(
        "--device", choices=["cpu", "cuda"], help="Device to train on"
    )

    training_group.add_argument(
        "--num-epochs", type=int, help="Number of maximum epochs."
    )
    if quantiles:
        training_group.add_argument(
            "--num-likelihood-samples",
            type=int,
            default=10,
            help=(
                "Number of latent GP samples used to estimate expected log "
                "likelihoods during training and validation."
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
    training_group.add_argument(
        "--cholesky-jitter",
        type=float,
        default=DEFAULT_CHOLESKY_JITTER,
        help=(
            "Initial diagonal jitter used by Cholesky factorizations. GPyTorch "
            "retries with successively larger values when needed."
        ),
    )
    training_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Maximum total gradient norm before each optimizer step.",
    )

    hpo_group.add_argument(
        "--optimize-hyperparameters",
        nargs="+",
        choices=tuple(hyperparameter_names),
        default=(),
        help=(
            "Hyperparameters to optimize with Optuna when validation data is "
            "provided; all others use their defaults."
        ),
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
        "--storage", type=str, default=None, help="Optuna storage URL."
    )
    hpo_group.add_argument(
        "--study-name", type=str, default=None, help="Optuna study name."
    )
    return parser


def validate_train_args(
    parser, args, logger, *, require_storage_without_validation=True
):
    has_validation = args.Xval is not None or args.yval is not None
    if (args.Xval is None) != (args.yval is None):
        parser.error("Xval and yval must be provided together.")
    if has_validation and args.num_epochs is None:
        parser.error("--num-epochs is required when validation data is provided.")
    if (
        not has_validation
        and require_storage_without_validation
        and args.storage is None
    ):
        parser.error("--storage is required when validation data is not provided.")
    if has_validation and args.num_epochs <= 0:
        parser.error("--num-epochs must be positive.")
    if hasattr(args, "num_likelihood_samples") and args.num_likelihood_samples <= 0:
        parser.error("--num-likelihood-samples must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.min_learning_rate < 0 or args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate must be between 0 and --learning-rate.")
    if not np.isfinite(args.cholesky_jitter) or args.cholesky_jitter <= 0:
        parser.error("--cholesky-jitter must be finite and positive.")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be finite and positive.")
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
    if hasattr(args, "quantiles"):
        if len(args.quantiles) < 2:
            parser.error("--quantiles must contain at least two values.")
        if not all(np.isfinite(q) and 0 < q < 1 for q in args.quantiles):
            parser.error("--quantiles values must be finite and in (0, 1).")
        if any(
            left >= right for left, right in zip(args.quantiles, args.quantiles[1:])
        ):
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

    device = torch.device(
        args.device
        if args.device is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
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
    else:
        early_stopping_patience = 0
        lr_scheduler_patience = 0
        pruning_warmup = 0
        pruning_patience = 0
        pruning_interval = 0
    return TrainingControls(
        has_validation=has_validation,
        device=device,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_patience=lr_scheduler_patience,
        pruning_warmup=pruning_warmup,
        pruning_patience=pruning_patience,
        pruning_interval=pruning_interval,
    )


def suggest_prior_hyperparameters(trial, defaults, optimized, args):
    hyperparameters = defaults.copy()
    if "noise_prior_loc" in optimized:
        hyperparameters["noise_prior_loc"] = trial.suggest_float(
            "noise_prior_loc", args.noise_prior_loc_min, args.noise_prior_loc_max
        )
    if "noise_prior_scale" in optimized:
        hyperparameters["noise_prior_scale"] = trial.suggest_float(
            "noise_prior_scale", args.prior_scale_min, args.prior_scale_max, log=True
        )
    if "lengthscale_prior_loc" in optimized:
        hyperparameters["lengthscale_prior_loc"] = trial.suggest_float(
            "lengthscale_prior_loc",
            args.lengthscale_prior_loc_min,
            args.lengthscale_prior_loc_max,
        )
    if "lengthscale_prior_scale" in optimized:
        hyperparameters["lengthscale_prior_scale"] = trial.suggest_float(
            "lengthscale_prior_scale",
            args.prior_scale_min,
            args.prior_scale_max,
            log=True,
        )
    return hyperparameters


def prior_baseline(args):
    return {
        "noise_prior_loc": float(
            np.clip(
                PRIOR_HYPERPARAMETER_BASELINES["noise_prior_loc"],
                args.noise_prior_loc_min,
                args.noise_prior_loc_max,
            )
        ),
        "noise_prior_scale": float(
            np.clip(
                PRIOR_HYPERPARAMETER_BASELINES["noise_prior_scale"],
                args.prior_scale_min,
                args.prior_scale_max,
            )
        ),
        "lengthscale_prior_loc": float(
            np.clip(
                PRIOR_HYPERPARAMETER_BASELINES["lengthscale_prior_loc"],
                args.lengthscale_prior_loc_min,
                args.lengthscale_prior_loc_max,
            )
        ),
        "lengthscale_prior_scale": float(
            np.clip(
                PRIOR_HYPERPARAMETER_BASELINES["lengthscale_prior_scale"],
                args.prior_scale_min,
                args.prior_scale_max,
            )
        ),
    }


def reuse_duplicate_trial(trial, logger):
    completed_trials = trial.study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    for completed_trial in reversed(completed_trials):
        if completed_trial.params != trial.params:
            continue
        for name, value in completed_trial.user_attrs.items():
            if name != "duplicate_of":
                trial.set_user_attr(name, value)
        source_trial_number = completed_trial.user_attrs.get(
            "duplicate_of", completed_trial.number
        )
        trial.set_user_attr("duplicate_of", source_trial_number)
        logger.info(
            "Trial %d reuses trial %d for duplicate parameters %s.",
            trial.number,
            source_trial_number,
            trial.params,
        )
        return completed_trial.value
    return None


def _likelihood_samples(num_likelihood_samples):
    if num_likelihood_samples is None:
        return nullcontext()
    return gpytorch.settings.num_likelihood_samples(num_likelihood_samples)


def _cholesky_jitter(value):
    """Apply one explicit jitter policy to all GPyTorch Cholesky paths.

    Variational strategies may promote float32 covariance matrices to float64
    before factorization. Setting both dtype-specific values avoids silently
    falling back to the much smaller float64 default in that path.
    """
    values = {"float_value": value, "double_value": value, "half_value": value}
    return (
        gpytorch.settings.cholesky_jitter(**values),
        gpytorch.settings.variational_cholesky_jitter(**values),
    )


def _backward_and_step(loss, parameters, optimizer, max_grad_norm):
    loss.backward()
    return _clip_and_step(parameters, optimizer, max_grad_norm)


def _clip_and_step(parameters, optimizer, max_grad_norm):
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
        error_if_nonfinite=False,
    )
    if not torch.isfinite(grad_norm):
        return False
    optimizer.step()
    return True


def _validation_expected_log_prob(
    likelihood,
    y,
    function_dist,
    *,
    use_data_independent_samples,
    jitter,
):
    """Evaluate pointwise likelihoods without sampling the full joint GP.

    The expected log likelihood is pointwise. Remove covariance across data
    points while retaining the multitask covariance within each point, avoiding
    an unnecessary ``(N * Q)^2`` joint covariance root decomposition.
    """
    if not use_data_independent_samples:
        return likelihood.expected_log_prob(y, function_dist)

    if isinstance(function_dist.lazy_covariance_matrix, BlockInterleavedLinearOperator):
        # IndependentMultitaskVariationalStrategy represents task independence
        # with interleaved covariance blocks. Extracting pointwise task
        # covariances through MultitaskMultivariateNormal's advanced indexing
        # can produce out-of-bounds CUDA indices for large validation batches.
        # These task covariances are diagonal by construction, so build the
        # equivalent data-independent distribution directly from the variance.
        function_dist = MultivariateNormal(
            function_dist.mean,
            DiagLinearOperator(function_dist.variance).add_jitter(jitter_val=jitter),
        )
    else:
        # GPyTorch's lazy advanced-indexing path can generate invalid CUDA
        # indices for LMC covariance operators, even for small data chunks.
        # Materialize only the bounded chunk and take its diagonal data blocks
        # with tensor views instead. This retains covariance between tasks at
        # each observation while discarding covariance between observations.
        num_data, num_tasks = function_dist.mean.shape[-2:]
        full_covar = function_dist.covariance_matrix
        if function_dist._interleaved:
            task_covars = full_covar.reshape(
                *full_covar.shape[:-2], num_data, num_tasks, num_data, num_tasks
            ).diagonal(dim1=-4, dim2=-2)
        else:
            task_covars = full_covar.reshape(
                *full_covar.shape[:-2], num_tasks, num_data, num_tasks, num_data
            ).diagonal(dim1=-3, dim2=-1)
        task_covars = task_covars.movedim(-1, -3).contiguous()
        function_dist = MultivariateNormal(
            function_dist.mean,
            to_linear_operator(task_covars).add_jitter(jitter_val=jitter),
        )
    return likelihood.expected_log_prob(y, function_dist)


def _training_minibatches(X, y, batch_size):
    """Yield one shuffled, complete pass over the observation dimension."""
    num_data = X.shape[-2]
    if batch_size is None or batch_size >= num_data:
        yield X, y
        return

    indices = torch.randperm(num_data, device=X.device)
    for batch_indices in indices.split(batch_size):
        yield (
            X.index_select(-2, batch_indices),
            y.index_select(-1, batch_indices),
        )


def _validation_loss(
    *,
    likelihood,
    model,
    X,
    y,
    reduce_dims,
    batch_size,
    use_data_independent_samples,
    jitter,
):
    """Evaluate the complete validation set in bounded-memory chunks."""
    num_data = X.shape[-2]
    chunk_size = num_data if batch_size is None else min(batch_size, num_data)
    weighted_loss = torch.zeros((), dtype=X.dtype, device=X.device)
    for start in range(0, num_data, chunk_size):
        stop = min(start + chunk_size, num_data)
        function_dist = model(X[..., start:stop, :])
        log_prob = _validation_expected_log_prob(
            likelihood,
            y[..., start:stop],
            function_dist,
            use_data_independent_samples=use_data_independent_samples,
            jitter=jitter,
        )
        chunk_loss = -log_prob.mean(dim=reduce_dims).mean()
        weighted_loss += chunk_loss * (stop - start)
    return weighted_loss / num_data


def fit_trial(
    *,
    likelihood,
    model,
    mll,
    parameters,
    Xtrain,
    ytrain,
    Xval,
    yval,
    args,
    controls,
    logger,
    validation_reduce_dims,
    trial=None,
    num_likelihood_samples=None,
    batch_size=None,
    return_training_details=False,
):
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=controls.lr_scheduler_patience,
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
        train_loss = torch.zeros((), dtype=Xtrain.dtype, device=Xtrain.device)
        for Xbatch, ybatch in _training_minibatches(Xtrain, ytrain, batch_size):
            batch_weight = Xbatch.shape[-2] / Xtrain.shape[-2]
            cholesky_context, variational_cholesky_context = _cholesky_jitter(
                args.cholesky_jitter
            )
            with (
                cholesky_context,
                variational_cholesky_context,
                _likelihood_samples(num_likelihood_samples),
            ):
                batch_loss = -mll(model(Xbatch), ybatch).mean()
            if not torch.isfinite(batch_loss):
                save_trial_progress("non_finite_train_loss")
                if trial is not None:
                    raise optuna.TrialPruned("Non-finite training loss.")
                raise RuntimeError("Non-finite training loss during final fit.")
            (batch_loss * batch_weight).backward()
            train_loss += batch_loss.detach() * batch_weight
        if not _clip_and_step(parameters, optimizer, args.max_grad_norm):
            save_trial_progress("non_finite_gradient")
            if trial is not None:
                raise optuna.TrialPruned("Non-finite gradient.")
            raise RuntimeError("Non-finite gradient during final fit.")

        cholesky_context, variational_cholesky_context = _cholesky_jitter(
            args.cholesky_jitter
        )
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            cholesky_context,
            variational_cholesky_context,
            _likelihood_samples(num_likelihood_samples),
        ):
            likelihood.eval()
            model.eval()
            val_loss = _validation_loss(
                likelihood=likelihood,
                model=model,
                X=Xval,
                y=yval,
                reduce_dims=validation_reduce_dims,
                batch_size=batch_size,
                use_data_independent_samples=num_likelihood_samples is not None,
                jitter=args.cholesky_jitter,
            )

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
            lr_reductions.append([epoch + 1, current_lr])

        if current_val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if trial is not None:
            trial.report(current_val_loss, step=epoch)
            if trial.should_prune():
                save_trial_progress("pruned")
                raise optuna.TrialPruned()

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Trial %s, epoch %d: train loss %.6f, validation loss %.6f, "
                "noise %.2e",
                trial.number if trial is not None else "best",
                epoch + 1,
                train_loss.item(),
                current_val_loss,
                likelihood.noise.mean().item(),
            )

        if epochs_without_improvement >= controls.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    else:
        stop_reason = "max_epochs"

    save_trial_progress(stop_reason)
    if return_training_details:
        return best_val_loss, best_epoch, lr_reductions
    return best_val_loss


def fit_all_data(
    *,
    likelihood,
    model,
    mll,
    parameters,
    X,
    y,
    learning_rate,
    cholesky_jitter,
    max_grad_norm,
    num_epochs,
    lr_reductions,
    logger,
    num_likelihood_samples=None,
    batch_size=None,
):
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    lr_reductions_by_epoch = {
        int(completed_epoch): float(new_lr) for completed_epoch, new_lr in lr_reductions
    }
    for epoch in range(num_epochs):
        likelihood.train()
        model.train()
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=X.dtype, device=X.device)
        for Xbatch, ybatch in _training_minibatches(X, y, batch_size):
            batch_weight = Xbatch.shape[-2] / X.shape[-2]
            cholesky_context, variational_cholesky_context = _cholesky_jitter(
                cholesky_jitter
            )
            with (
                cholesky_context,
                variational_cholesky_context,
                _likelihood_samples(num_likelihood_samples),
            ):
                batch_loss = -mll(model(Xbatch), ybatch).mean()
            if not torch.isfinite(batch_loss):
                raise RuntimeError("Non-finite training loss during final fit.")
            (batch_loss * batch_weight).backward()
            loss += batch_loss.detach() * batch_weight
        if not _clip_and_step(parameters, optimizer, max_grad_norm):
            raise RuntimeError("Non-finite gradient during final fit.")

        completed_epoch = epoch + 1
        if completed_epoch in lr_reductions_by_epoch:
            new_lr = lr_reductions_by_epoch[completed_epoch]
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = new_lr

        if completed_epoch % 100 == 0:
            logger.info(
                "Final training, epoch %d/%d: loss %.6f",
                completed_epoch,
                num_epochs,
                loss.item(),
            )


def initialize_optuna_storage(storage):
    """Initialize file-backed SQLite storage without a cross-process DDL race."""
    if not isinstance(storage, str):
        return storage

    url = make_url(storage)
    if url.get_backend_name() != "sqlite" or url.database in (None, ":memory:"):
        return storage

    database_path = pathlib.Path(url.database).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = database_path.with_name(f".{database_path.name}.init.lock")
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return optuna.storages.RDBStorage(storage)


def optimize_or_load_study(
    *,
    args,
    controls,
    objective,
    optimized_hyperparameters,
    baseline_params,
):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.study_name if args.study_name is not None else args.out.stem
    storage = initialize_optuna_storage(args.storage)
    if not controls.has_validation:
        return optuna.load_study(study_name=study_name, storage=storage)

    median_pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=controls.pruning_warmup,
        interval_steps=controls.pruning_interval,
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
            patience=controls.pruning_patience,
            min_delta=args.early_stopping_min_delta,
        ),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(
            {
                name: value
                for name, value in baseline_params.items()
                if name in optimized_hyperparameters
            }
        )
    study.optimize(
        objective,
        n_trials=args.n_trials if optimized_hyperparameters else 1,
        catch=(torch.linalg.LinAlgError, NotPSDError),
    )
    return study


def select_best_trial(study, defaults, optimized_hyperparameters, controls, args):
    best_trial = study.best_trial
    selected_hyperparameters = (
        optimized_hyperparameters if controls.has_validation else set(best_trial.params)
    )
    best_hyperparameters = {
        name: (
            best_trial.params.get(name, default)
            if name in selected_hyperparameters
            else default
        )
        for name, default in defaults.items()
    }
    if "best_epoch" not in best_trial.user_attrs:
        study_name = args.study_name if args.study_name is not None else args.out.stem
        raise RuntimeError(
            f"Best trial {best_trial.number} in study {study_name!r} has no "
            "'best_epoch' user attribute."
        )
    best_epoch = max(1, int(best_trial.user_attrs["best_epoch"]))
    if controls.has_validation:
        best_epoch = min(best_epoch, args.num_epochs)
    return (
        best_trial,
        best_hyperparameters,
        best_epoch,
        best_trial.user_attrs.get("lr_reductions", []),
    )
