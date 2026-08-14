import torch

__all__ = [
    "save_prior_mean",
    "save_gpr",
    "save_gpqr",
]


def save_prior_mean(model, path):
    data = dict(
        model=dict(
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
            args=dict(dim=X_scaler.dim, batch_shape=X_scaler.batch_shape),
            state_dict=X_scaler.state_dict(),
        ),
        y_scaler=dict(
            args=dict(dim=y_scaler.dim, batch_shape=y_scaler.batch_shape),
            state_dict=y_scaler.state_dict(),
        ),
        likelihood=dict(
            args=dict(
                noise_prior_loc=model.likelihood.noise_prior_loc,
                noise_prior_scale=model.likelihood.noise_prior_scale,
                batch_shape=model.likelihood.batch_shape,
            ),
            state_dict=model.likelihood.state_dict(),
        ),
        model=dict(
            args=dict(
                train_x=model.train_inputs[0],
                train_y=model.train_targets,
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


def save_gpqr(
    train_x,
    train_y,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    inducing_points,
    quantiles,
    num_lower_quantiles,
    num_latents,
    path,
):
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "X_scaler_state_dict": X_scaler.state_dict(),
            "y_scaler_state_dict": y_scaler.state_dict(),
            "mean_state_dict": mean.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "model_state_dict": model.state_dict(),
            "inducing_points": inducing_points,
            "quantiles": quantiles,
            "num_lower_quantiles": num_lower_quantiles,
            "num_latents": num_latents,
        },
        path,
    )
