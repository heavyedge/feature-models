import torch

__all__ = [
    "PriorMean",
]


class PriorMean(torch.nn.Module):
    """Prior means for three independently batched outputs.

    Input X: ``(*K, N, D)`` with ``[Rgt, Ca, cos_theta, ...]`` features.
    Output y: ``(*K, 3, N)`` ordered as ``H``, ``phi_1``, ``phi_3``.
    """

    output_names = ("H", "phi_1", "phi_3")

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        batch_shape = torch.Size(batch_shape)

        H_params = torch.tensor([0.22, -0.43, 0.77]).repeat(*batch_shape, 1)
        phi_params = torch.zeros(*batch_shape, 2, 2)
        self.params = torch.nn.ParameterDict(
            dict(H_params=H_params, phi_params=phi_params)
        )
        self.batch_shape = batch_shape

    def forward(self, x):
        Rgt = x[..., 0]  # (*K, N)
        Ca = x[..., 1]  # (*K, N)
        cos_theta = x[..., 2]  # (*K, N)

        a = self.params["H_params"][..., 0].unsqueeze(-1)
        b = self.params["H_params"][..., 1].unsqueeze(-1)
        c = self.params["H_params"][..., 2].unsqueeze(-1)
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))
        H_model = Rgt / E
        corrected_H_model = H_model.clamp_min(1.0)
        H_model = H_model + (corrected_H_model - H_model).detach()

        phi_a = self.params["phi_params"][..., :, 0].unsqueeze(-1)
        phi_b = self.params["phi_params"][..., :, 1].unsqueeze(-1)
        phi_model = phi_a * Rgt.unsqueeze(-2) + phi_b  # (*K, 2, N)

        return torch.cat((H_model.unsqueeze(-2), phi_model), dim=-2)
