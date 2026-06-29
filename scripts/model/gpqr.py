import gpytorch
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.means import CenterGapMean
from gpytorch_qr.models import CenterGapQuantileGP
from gpytorch_qr.variational import CGLmcVariationalStrategy

from .prior import PriorMean_H, Unscaler

__all__ = [
    "CgLmcMtgpqr_H",
    "CgIndependentMtgpqr_phi",
]


class CgLmcMtgpqr_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CGLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                PriorMean_H(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgIndependentMtgpqr_phi(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(
                    batch_shape=torch.Size([*batch_shape, 1]),
                ),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)
