import numpy as np
import torch
from gpytorch.mlls import VariationalELBO

__all__ = [
    "quantile_crossing",
]


def quantile_crossing(
    X_train,
    y_train,
    X_preds,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = VariationalELBO(likelihood, model, num_data=len(y_train))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=learning_rate,
    )

    crossing_rates = np.empty((len(X_preds), n_epochs))
    mean_crossings = np.empty((len(X_preds), n_epochs))
    max_crossings = np.empty((len(X_preds), n_epochs))
    for j in range(n_epochs):
        model.train()
        likelihood.train()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            for i, X_pred in enumerate(X_preds):
                output = model.mean_quantiles_delta(X_pred)
                quantile_diff = output.diff(axis=-1)
                crossing = quantile_diff < 0

                crossing_rates[i, j] = (
                    crossing.count_nonzero() / quantile_diff.numel()
                ).item()
                mean_crossings[i, j] = (
                    -quantile_diff[crossing].sum() / quantile_diff.numel()
                ).item()
                max_crossings[i, j] = (-quantile_diff).clip(0).max().item()

        logger(f"Epoch {j+1}/{n_epochs}, Loss: {loss.item():.4f}")

    return crossing_rates, mean_crossings, max_crossings
