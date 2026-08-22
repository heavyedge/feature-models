import torch

__all__ = [
    "save_prior_mean",
    "save_gpr",
    "save_gpqr",
]


def _prior_args(module, name):
    loc = getattr(module, f"{name}_prior_loc", None)
    scale = getattr(module, f"{name}_prior_scale", None)
    if loc is None or scale is None:
        loc = scale = None
    return {
        f"{name}_prior_loc": loc,
        f"{name}_prior_scale": scale,
    }


def save_prior_mean(model, path):
    data = dict(
        model=dict(
            type=model.__class__.__name__,
            args=dict(batch_shape=model.batch_shape),
            state_dict=model.state_dict(),
        )
    )
    torch.save(
        data,
        path,
    )


def save_gpr(
    X_scaler,
    y_scaler,
    model,
    path,
):
    data = dict(
        X_scaler=dict(
            type=X_scaler.__class__.__name__,
            args=dict(dim=X_scaler.dim, batch_shape=X_scaler.batch_shape),
            state_dict=X_scaler.state_dict(),
        ),
        y_scaler=dict(
            type=y_scaler.__class__.__name__,
            args=dict(dim=y_scaler.dim, batch_shape=y_scaler.batch_shape),
            state_dict=y_scaler.state_dict(),
        ),
        likelihood=dict(
            type=model.likelihood.__class__.__name__,
            args=dict(
                **_prior_args(model.likelihood, "noise"),
                batch_shape=model.likelihood.batch_shape,
            ),
            state_dict=model.likelihood.state_dict(),
        ),
        model=dict(
            type=model.__class__.__name__,
            args=dict(
                train_x=model.train_inputs[0],
                train_y=model.train_targets,
                **_prior_args(model, "lengthscale"),
                batch_shape=model.batch_shape,
            ),
            state_dict=model.state_dict(),
        ),
    )
    torch.save(
        data,
        path,
    )


def save_gpqr(
    quantiles,
    X_scaler,
    y_scaler,
    likelihood,
    model,
    path,
):
    data = dict(
        quantiles=quantiles,
        X_scaler=dict(
            type=X_scaler.__class__.__name__,
            args=dict(dim=X_scaler.dim, batch_shape=X_scaler.batch_shape),
            state_dict=X_scaler.state_dict(),
        ),
        y_scaler=dict(
            type=y_scaler.__class__.__name__,
            args=dict(dim=y_scaler.dim, batch_shape=y_scaler.batch_shape),
            state_dict=y_scaler.state_dict(),
        ),
        likelihood=dict(
            type=likelihood.__class__.__name__,
            args=dict(
                quantile_levels=likelihood.quantile_levels,
                central_quantile_idx=likelihood.central_quantile_idx,
                **_prior_args(likelihood, "noise"),
                batch_shape=likelihood.batch_shape,
            ),
            state_dict=likelihood.state_dict(),
        ),
        model=dict(
            type=model.__class__.__name__,
            args=dict(
                inducing_points=model.inducing_points,
                num_quantiles=model.num_quantiles[0],
                num_lower_quantiles=model.num_lower_quantiles[0],
                num_latents=model.num_latents,
                **_prior_args(model, "lengthscale"),
                batch_shape=model.batch_shape,
            ),
            state_dict=model.state_dict(),
        ),
    )
    torch.save(
        data,
        path,
    )
