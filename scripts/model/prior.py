import torch

__all__ = [
    "PriorMean_H",
    "PriorMean_b",
    "PriorMean_phi",
]


class PriorMean_H(torch.nn.Module):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, cos_theta, ...].
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape
        self.params = torch.nn.ParameterDict(
            {
                "a": torch.nn.Parameter(torch.full(batch_shape, 1.0)),
                "b": torch.nn.Parameter(torch.full(batch_shape, 0.0)),
                "c": torch.nn.Parameter(torch.full(batch_shape, 0.0)),
            }
        )

    def forward(self, x):
        Rgt = x[..., 0]  # (*B, N)
        Ca = x[..., 1]  # (*B, N)
        cos_theta = x[..., 2]  # (*B, N)

        a = self.params["a"].unsqueeze(-1)  # (*B, 1)
        b = self.params["b"].unsqueeze(-1)  # (*B, 1)
        c = self.params["c"].unsqueeze(-1)  # (*B, 1)

        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model  # (*B, N)


class PriorMean_b(torch.nn.Module):
    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, x):
        N = x.shape[-2]
        return torch.zeros(*self.batch_shape, N, device=x.device)


class PriorMean_phi(torch.nn.Module):
    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, x):
        N = x.shape[-2]
        return torch.zeros(*self.batch_shape, N, device=x.device)
