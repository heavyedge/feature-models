import gpytorch
import numpy as np
import torch
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr import MTGPQR, CenterGapGP, CenterGapLmcVariationalStrategy

__all__ = [
    "Scaler",
    "Unscaler",
    "PriorMean_H",
    "MTGP_H",
    "MTGPQR_H",
    "save_mtgpqr",
    "load_mtgpqr",
    "quantile_interpolation",
]


class Scaler(torch.nn.Module):
    """Min-max-scaling."""

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        return x * self.X_scale + self.X_min


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling."""

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        return (x - self.X_min) / self.X_scale


class PriorMean_H(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self):
        super().__init__()
        self.register_parameter(
            "model_parameters",
            torch.nn.Parameter(torch.tensor([0.25, -0.27, -1.0, -1.0]).float()),
        )

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        st = x[..., 2]

        a, b, c, d = self.model_parameters
        lamda = (a * Rgt + b) * Ca**c * st**d
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model


class MTGP_H(CenterGapGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale,
        X_min,
    ):
        N, D = inducing_points.size()
        variational_strategy = CenterGapLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                CholeskyVariationalDistribution(
                    N,
                    batch_shape=torch.Size([num_latents]),
                ),
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
            latent_dim=-1,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        center_mean = torch.nn.Sequential(Unscaler(X_scale, X_min), PriorMean_H())
        gap_mean = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents - 1])
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=D,
                batch_shape=torch.Size([num_latents]),
            ),
            batch_shape=torch.Size([num_latents]),
        )
        super().__init__(variational_strategy, center_mean, gap_mean, covar_module)


class MTGPQR_H(MTGPQR):
    def __init__(self, inducing_points, X_scale=None, X_min=None):
        _, D = inducing_points.size()
        taus = torch.tensor([0.05, 0.5, 0.95])
        central_tau = taus[(taus - 0.5).abs().argmin()]
        num_lower_quantiles = len(taus[taus < central_tau])
        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        gp = MTGP_H(
            inducing_points=inducing_points,
            num_quantiles=len(taus),
            num_lower_quantiles=num_lower_quantiles,
            num_latents=9,
            num_lower_latents=4,
            X_scale=X_scale,
            X_min=X_min,
        )
        super().__init__(taus, gp)
        self.inducing_points = inducing_points
        self.taus = taus
        self.scaler = Scaler(X_scale, X_min)

    def forward(self, x):
        x_scaled = self.scaler(x)
        return super().forward(x_scaled)


def save_mtgpqr(model, path):
    torch.save(
        {
            "inducing_points": model.inducing_points,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_mtgpqr(model_class, path):
    checkpoint = torch.load(path)
    model = model_class(checkpoint["inducing_points"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


def quantile_interpolation(q_values, q_levels, threshold):
    idx = np.array([np.searchsorted(row, threshold) for row in q_values])
    idx_clamped = np.clip(idx, 1, len(q_levels) - 1)

    rows = np.arange(len(q_values))
    x0 = q_values[rows, idx_clamped - 1]
    x1 = q_values[rows, idx_clamped]
    y0 = q_levels[idx_clamped - 1]
    y1 = q_levels[idx_clamped]
    probs = y0 + (threshold - x0) * (y1 - y0) / (x1 - x0)

    probs = np.where(idx == 0, 0.0, probs)
    probs = np.where(idx == len(q_levels), 1.0, probs)
    return probs
