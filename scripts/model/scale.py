import torch

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
]


class MinMaxScaler(torch.nn.Module):
    """Min-max-scaling.

    Parameters
    ----------
    dim : int
        Dimension of the input data.
    batch_shape : torch.Size
        Shape of the batch dimension.
    """

    def __init__(self, dim, batch_shape=torch.Size()):
        super().__init__()
        self.dim = dim
        self.batch_shape = batch_shape

        self.register_buffer("X_min", torch.zeros(batch_shape + torch.Size([dim])))
        self.register_buffer("X_max", torch.zeros(batch_shape + torch.Size([dim])))

    def forward(self, x):
        if self.training:
            self.register_buffer("X_min", x.min(dim=-2).values)
            self.register_buffer("X_max", x.max(dim=-2).values)
        X_scale = self.X_max - self.X_min
        return (x - self.X_min.unsqueeze(-2)) / X_scale.unsqueeze(-2)

    def inverse_transform(self, x):
        X_scale = self.X_max - self.X_min
        return x * X_scale.unsqueeze(-2) + self.X_min.unsqueeze(-2)


class StandardScaler(torch.nn.Module):
    """Standard scaling.

    Parameters
    ----------
    dim : int
        Dimension of the input data.
    batch_shape : torch.Size
        Shape of the batch dimension.
    """

    def __init__(self, dim, batch_shape=torch.Size()):
        super().__init__()
        self.dim = dim
        self.batch_shape = batch_shape

        self.register_buffer("X_mean", torch.zeros(batch_shape + torch.Size([dim])))
        self.register_buffer("X_scale", torch.zeros(batch_shape + torch.Size([dim])))

    def forward(self, x):
        if self.training:
            self.register_buffer("X_mean", x.mean(dim=-2))
            self.register_buffer("X_scale", x.std(dim=-2))
        return (x - self.X_mean.unsqueeze(-2)) / self.X_scale.unsqueeze(-2)

    def inverse_transform(self, x):
        return x * self.X_scale.unsqueeze(-2) + self.X_mean.unsqueeze(-2)
