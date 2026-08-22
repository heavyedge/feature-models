import torch

__all__ = [
    "PriorMean_H",
    "PriorMean_phi",
]


class PriorMean_H(torch.nn.Module):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, cos_theta, ...].
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        # Initial values are heuristically chosen.
        a = torch.tensor(0.22).repeat(*batch_shape, 1)  # (*B, 1)
        b = torch.tensor(-0.43).repeat(*batch_shape, 1)  # (*B, 1)
        c = torch.tensor(0.77).repeat(*batch_shape, 1)  # (*B, 1)
        self.params = torch.nn.ParameterDict(dict(a=a, b=b, c=c))
        self.batch_shape = batch_shape

    def forward(self, x):
        Rgt = x[..., 0]  # (*B, N)
        Ca = x[..., 1]  # (*B, N)
        cos_theta = x[..., 2]  # (*B, N)

        a = self.params["a"]  # (*B, 1)
        b = self.params["b"]  # (*B, 1)
        c = self.params["c"]  # (*B, 1)

        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = model.clamp_min(1.0)
        # Preserve the physical lower bound in the forward pass while keeping
        # gradients alive when the initial model is below 1 for every sample.
        return model + (corrected_model - model).detach()  # (*B, N)


class PriorMean_phi(torch.nn.Module):
    """Heuristical linear model.

    Input X must be [Rgt, Ca, cos_theta, ...].
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        phi_params = torch.zeros(*batch_shape, 2)  # (*B, 2)
        self.params = torch.nn.ParameterDict(dict(phi_params=phi_params))
        self.batch_shape = batch_shape

    def forward(self, x):
        Rgt = x[..., 0]  # (*B, N)

        design_matrix = torch.stack((Rgt, torch.ones_like(Rgt)), dim=-1)  # (*B, N, 2)
        phi_params = self.params["phi_params"].unsqueeze(-1)  # (*B, 2, 1)
        return torch.matmul(design_matrix, phi_params).squeeze(-1)  # (*B, N)
