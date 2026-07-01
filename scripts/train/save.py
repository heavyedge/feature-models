import torch

__all__ = [
    "save_model",
    "save_gpr2",
]


def save_model(
    train_x,
    train_y,
    model,
    likelihood,
    scaler,
    inducing_points,
    quantiles,
    num_lower_quantiles,
    num_latents,
    num_lower_latents,
    path,
):
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "scaler": scaler,
            "inducing_points": inducing_points,
            "quantiles": quantiles,
            "num_lower_quantiles": num_lower_quantiles,
            "num_latents": num_latents,
            "num_lower_latents": num_lower_latents,
        },
        path,
    )


def save_gpr2(
    train_x,
    train_y,
    X_scaler,
    y_scaler,
    mean,
    likelihood,
    model,
    path,
):
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "X_scaler_state_dict": X_scaler.state_dict(),
            "y_scaler_state_dict": y_scaler.state_dict(),
            "mean_state_dict": mean.state_dict(),
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
        },
        path,
    )
