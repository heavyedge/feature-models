import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.models import CenterGapQuantileGP

__all__ = [
    "CenterGapMTGPQR_H",
]


class CenterGapMTGPQR_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
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
        super().__init__(
            variational_strategy, mean, covar, [num_quantiles], [num_lower_quantiles]
        )

    quantiles = CenterGapQuantileGP.mean_quantiles_delta
