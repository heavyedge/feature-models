import numpy as np
import torch
from gpytorch.mlls import VariationalELBO

__all__ = [
    "quantile_crossing",
]


def quantile_crossing(
    X_train,  # (*B, N_train, D)
    y_train,  # (*B, N_train)
    X_preds,  # list of (*B, N_pred, D)
    X_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mean.eval()
    mll = VariationalELBO(likelihood, model, num_data=y_train.shape[-1])
    optimizer = torch.optim.Adam(
        list(X_scaler.parameters())
        + list(y_scaler.parameters())
        + list(model.parameters())
        + list(likelihood.parameters()),
        lr=learning_rate,
    )

    crossing_rates = np.empty((len(X_preds), n_epochs))
    mean_crossings = np.empty((len(X_preds), n_epochs))
    max_crossings = np.empty((len(X_preds), n_epochs))
    for i in range(n_epochs):
        X_scaler.train()
        y_scaler.train()
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        train_x_scaled = X_scaler(X_train)
        with torch.no_grad():
            train_mean = mean(X_train)
        train_res = y_scaler((y_train - train_mean).unsqueeze(-1)).squeeze(-1)
        train_output = model(train_x_scaled)
        train_loss = -mll(train_output, train_res)
        train_loss.mean().backward()
        optimizer.step()

        X_scaler.eval()
        y_scaler.eval()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            for j, X_pred in enumerate(X_preds):
                output = model.quantiles(X_scaler(X_pred))
                quantile_diff = output.diff(axis=-1)
                crossing = quantile_diff < 0

                crossing_rates[j, i] = (
                    crossing.count_nonzero() / quantile_diff.numel()
                ).item()
                mean_crossings[j, i] = (
                    -quantile_diff[crossing].sum() / quantile_diff.numel()
                ).item()
                max_crossings[j, i] = (-quantile_diff).clip(0).max().item()

        logger(f"Epoch {i+1}/{n_epochs}, Loss: {train_loss.mean().item():.4f}")

    return crossing_rates, mean_crossings, max_crossings
