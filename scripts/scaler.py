import torch

__all__ = [
    "Unscaler",
]


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling.

    Parameters
    ----------
    X_scale, X_min: torch.Tensor in shape (*B, D)
    """

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        # x: (*B, 1, N, D)
        # BELOW IS CORRECT FOR MinMaxScaler (different from StandardScaler)
        X_min = self.X_min[..., None, None, :]
        X_scale = self.X_scale[..., None, None, :]
        x_unscaled = (x - X_min) / X_scale
        return x_unscaled.view_as(x)
