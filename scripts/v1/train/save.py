import torch

__all__ = [
    "save_gpr",
]


def save_gpr(
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
            "likelihood_state_dict": likelihood.state_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
