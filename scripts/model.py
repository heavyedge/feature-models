import gpytorch
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.likelihoods import MultitaskCenterGapQuantileGPLikelihood
from gpytorch_qr.means import CenterGapMean
from gpytorch_qr.models import CenterGapQuantileGP
from gpytorch_qr.variational import CGBlkdiagLmcVariationalStrategy

__all__ = [
    "Unscaler",
    "PriorMean_H",
    "MTGPQR_H",
    "MTGPQR_phi",
    "save_model",
    "load_model",
]


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling."""

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        x_flattened = x.view(-1, x.shape[-1])
        x_unscaled = (x_flattened - self.X_min) / self.X_scale
        return x_unscaled.view_as(x)


class PriorMean_H(gpytorch.means.Mean):
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


class MTGPQR_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
    ):
        N, D = inducing_points.size()
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=torch.Size([num_latents]),
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
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(unscaler, PriorMean_H(batch_shape=torch.Size([1]))),
            ConstantMean(batch_shape=torch.Size([num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class MTGPQR_phi(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
    ):
        N, D = inducing_points.size()
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=torch.Size([num_latents]),
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
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(unscaler, ConstantMean(batch_shape=torch.Size([1]))),
            ConstantMean(batch_shape=torch.Size([num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )
        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


def save_model(
    model,
    likelihood,
    scaler,
    inducing_points,
    quantiles,
    num_lower_quantiles,
    num_latents,
    num_lower_latents,
    path,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "scaler": scaler,
            "inducing_points": inducing_points,
            "quantiles": quantiles,
            "num_lower_quantiles": num_lower_quantiles,
            "num_latents": num_latents,
            "num_lower_latents": num_lower_latents,
        },
        path,
    )


def load_model(model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = model_class(
        inducing_points=checkpoint["inducing_points"],
        num_quantiles=len(checkpoint["quantiles"]),
        num_lower_quantiles=checkpoint["num_lower_quantiles"],
        num_latents=checkpoint["num_latents"],
        num_lower_latents=checkpoint["num_lower_latents"],
    )
    likelihood = MultitaskCenterGapQuantileGPLikelihood(
        checkpoint["quantiles"],
        checkpoint["num_lower_quantiles"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    if device is not None:
        model.to(device)
        likelihood.to(device)
    scaler = checkpoint["scaler"]
    return model, likelihood, scaler
