import torch

__all__ = [
    "MinMaxScaler",
]


class MinMaxScaler(torch.nn.Module):
    """Min-max-scaling.

    Parameters
    ----------
    X_scale, X_min: torch.Tensor in shape (*B, D)
        Values from sklearn.preprocessing.MinMaxScaler.
    """

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        # x: (*B, N, D)
        # X_scale, X_min: (*B, D)
        # BELOW IS CORRECT FOR MinMaxScaler (different from StandardScaler)
        X_min = self.X_min.unsqueeze(-2)
        X_scale = self.X_scale.unsqueeze(-2)
        x_scaled = x * X_scale + X_min
        return x_scaled.view_as(x)
