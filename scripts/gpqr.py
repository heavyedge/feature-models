import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.means import CenterGapMean
from gpytorch_qr.models import CenterGapQuantileGP
from gpytorch_qr.variational import CGBlkdiagLmcVariationalStrategy

__all__ = [
    "Scaler",
    "Unscaler",
    "PriorMean_H",
    "CgLmcMtgpqr_H",
]


class Scaler(torch.nn.Module):
    """Min-max-scaling.

    Parameters
    ----------
    X_scale, X_mean: torch.Tensor in shape (*B, D)
    """

    def __init__(self, X_scale, X_mean):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_mean", X_mean)

    def forward(self, x):
        # x: (*B, N, D)
        return x * self.X_scale.unsqueeze(-2) + self.X_mean.unsqueeze(-2)


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling.

    Parameters
    ----------
    X_scale, X_mean: torch.Tensor in shape (*B, D)
    """

    def __init__(self, X_scale, X_mean):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_mean", X_mean)

    def forward(self, x):
        # x: (*B, 1, N, D)
        ret = (x - self.X_mean[..., None, None, :]) / self.X_scale[..., None, None, :]
        return ret


class PriorMean_H(Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a, b, c = 0.22, -0.43, 0.77  # From GPR prior
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model


class CgLmcMtgpqr_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_mean=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=batch_shape,
        )
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_mean is None:
            X_mean = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_mean=X_mean)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler, PriorMean_H(batch_shape=torch.Size([*batch_shape[:-1], 1]))
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape[:-1], num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)
        self.scaler = Scaler(X_scale=X_scale, X_mean=X_mean)

    def forward(self, x):
        x_scaled = self.scaler(x)
        return super().forward(x_scaled)
