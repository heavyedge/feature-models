import numpy as np
import torch
from gpytorch.mlls import VariationalELBO

__all__ = [
    "quantile_crossing",
]


def quantile_crossing(
    X_train,
    y_train,
    X_pred,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
):
    mll = VariationalELBO(likelihood, model, num_data=len(y_train))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=learning_rate,
    )

    crossing_rate, mean_crossing, max_crossing = [], [], []
    for _ in range(n_epochs):
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
            output = model.mean_quantiles_delta(X_pred)
            quantile_diff = output.diff(axis=-1)
            crossing = quantile_diff < 0

            crossing_rate.append(
                (crossing.count_nonzero() / quantile_diff.numel()).item()
            )
            mean_crossing.append(
                (-quantile_diff[crossing].sum() / quantile_diff.numel()).item()
            )
            max_crossing.append((-quantile_diff).clip(0).max().item())

    return np.array(crossing_rate), np.array(mean_crossing), np.array(max_crossing)
