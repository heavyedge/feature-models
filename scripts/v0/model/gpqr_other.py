import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.models import DirectQuantileGP

__all__ = [
    "DirectMTGPQR_H",
    "DirectMTGPQR_phi",
]


class DirectMTGPQR_H(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
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
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles,
            num_latents,
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

        super().__init__(variational_strategy, mean, covar)

    quantiles = DirectQuantileGP.mean_quantiles_delta


class DirectMTGPQR_phi(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
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
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles,
            num_latents,
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

        super().__init__(variational_strategy, mean, covar)

    quantiles = DirectQuantileGP.mean_quantiles_delta
