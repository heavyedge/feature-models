import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
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
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 0.5, 0.5] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([1, 0.5, 0.5] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint("raw_lengthscale", Interval(lower, upper))
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar)

    quantiles = DirectQuantileGP.mean_quantiles_delta


class DirectMTGPQR_phi(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_latents,
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
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 0.5, 0.5] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([1, 0.5, 0.5] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint("raw_lengthscale", Interval(lower, upper))
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar)

    quantiles = DirectQuantileGP.mean_quantiles_delta
