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
        self.register_buffer("a", torch.full(batch_shape, 0.22))
        self.register_buffer("b", torch.full(batch_shape, -0.43))
        self.register_buffer("c", torch.full(batch_shape, 0.77))

    def forward(self, x):
        Rgt = x[..., 0]  # (*B, N)
        Ca = x[..., 1]  # (*B, N)
        cos_theta = x[..., 2]  # (*B, N)

        a = self.a.unsqueeze(-1)  # (*B, 1)
        b = self.b.unsqueeze(-1)  # (*B, 1)
        c = self.c.unsqueeze(-1)  # (*B, 1)

        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = model.clamp_min(1.0)
        # Preserve the physical lower bound in the forward pass while keeping
        # gradients alive when the initial model is below 1 for every sample.
        return model + (corrected_model - model).detach()  # (*B, N)


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
