import torch

__all__ = [
    "PriorMean",
]


class PriorMean(torch.nn.Module):
    """Multi-output mean.

    Input X: [Rgt, Ca, cos_theta, ...]
    Output y: [H, phi_1, phi_3]

    H follows the modified Schmitt model, and
    phi_i follow a simple linear model.
    """

    output_names = ("H", "phi_1", "phi_3")

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        # Initial values are heuristically chosen.
        H_params = torch.tensor([0.22, -0.43, 0.77]).repeat(*batch_shape, 1)  # (*B, 3)
        phi_params = torch.zeros(*batch_shape, 2, 2)  # (*B, 2, 2)
        self.params = torch.nn.ParameterDict(
            dict(H_params=H_params, phi_params=phi_params)
        )
        self.batch_shape = batch_shape

    def forward(self, x):
        Rgt = x[..., 0]  # (*B, N)
        Ca = x[..., 1]  # (*B, N)
        cos_theta = x[..., 2]  # (*B, N)

        # H
        a = self.params["H_params"][..., 0].unsqueeze(-1)  # (*B, 1)
        b = self.params["H_params"][..., 1].unsqueeze(-1)  # (*B, 1)
        c = self.params["H_params"][..., 2].unsqueeze(-1)  # (*B, 1)
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))
        H_model = (Rgt / E).clamp_min(1.0).unsqueeze(-1)  # (*B, N, 1)

        # phis
        a = self.params["phi_params"][..., 0, :].unsqueeze(-2)  # (*B, 1, 2)
        b = self.params["phi_params"][..., 1, :].unsqueeze(-2)  # (*B, 1, 2)
        phi_model = a * Rgt[..., None] + b  # (*B, N, 2)

        return torch.cat([H_model, phi_model], dim=-1)  # (*B, N, 3)
