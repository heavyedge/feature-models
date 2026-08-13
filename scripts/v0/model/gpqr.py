import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.models import CenterGapQuantileGP
from gpytorch_qr.variational import CenterGapLMCVariationalStrategy

__all__ = [
    "CenterGapMTGPQR_H",
    "CenterGapMTGPQR_phi",
]


class CenterGapMTGPQR_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CenterGapLMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        mean = ConstantMean(batch_shape=full_batch_shape)
        ls_prior = LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale)
        covar = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=full_batch_shape,
                lengthscale_prior=ls_prior,
            ),
            batch_shape=full_batch_shape,
        )

        super().__init__(
            variational_strategy,
            mean,
            covar,
            [num_quantiles],
            [num_lower_quantiles],
        )

    quantiles = CenterGapQuantileGP.mean_quantiles_delta


class CenterGapMTGPQR_phi(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CenterGapLMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        mean = ConstantMean(batch_shape=full_batch_shape)
        ls_prior = LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale)
        covar = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=full_batch_shape,
                lengthscale_prior=ls_prior,
            ),
            batch_shape=full_batch_shape,
        )

        super().__init__(
            variational_strategy,
            mean,
            covar,
            [num_quantiles],
            [num_lower_quantiles],
        )

    quantiles = CenterGapQuantileGP.mean_quantiles_delta
