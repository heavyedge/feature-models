import torch

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
]


class MinMaxScaler(torch.nn.Module):
    """Min-max-scaling.

    Parameters
    ----------
    batch_shape : torch.Size
        Shape of the batch dimension.
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, x):
        if self.training:
            self.register_buffer("X_min", x.min(dim=-2).values)
            self.register_buffer("X_max", x.max(dim=-2).values)
        X_scale = self.X_max - self.X_min
        return (x - self.X_min.unsqueeze(-2)) / X_scale.unsqueeze(-2)


class StandardScaler(torch.nn.Module):
    """Standard scaling.

    Parameters
    ----------
    batch_shape : torch.Size
        Shape of the batch dimension.
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, x):
        if self.training:
            self.register_buffer("X_mean", x.mean(dim=-2))
            self.register_buffer("X_scale", x.std(dim=-2))
        return (x - self.X_mean.unsqueeze(-2)) / self.X_scale.unsqueeze(-2)
