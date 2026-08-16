import torch

__all__ = [
    "save_gpr",
]


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
        model=dict(
            type=model.__class__.__name__,
            args=dict(
                inducing_points=model.inducing_points,
                lengthscale_prior_loc=model.lengthscale_prior_loc,
                lengthscale_prior_scale=model.lengthscale_prior_scale,
                batch_shape=model.batch_shape,
            ),
            state_dict=model.state_dict(),
        ),
    )
    torch.save(
        data,
        path,
    )
