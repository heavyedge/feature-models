import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.models import CenterGapQuantileGP
from gpytorch_qr.variational import CenterGapLMCVariationalStrategy

__all__ = [
    "GPQR_Independent",
    "GPQR_LMC",
    "GPQR_CenterGapLMC",
]


class BaseGP(CenterGapQuantileGP):

    output_names = ("H", "phi_1", "phi_3")
    num_tasks = len(output_names)

    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_central_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.size()[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = self.construct_variational_strategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=sum(num_quantiles),
            num_latents=num_latents,
            num_central_latents=num_central_latents,
            num_quantiles=num_quantiles,
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
            variational_strategy, mean, covar, num_quantiles, num_lower_quantiles
        )

        self.num_latents = num_latents
        self.num_central_latents = num_central_latents
        self.batch_shape = batch_shape

    @staticmethod
    def construct_variational_strategy(
        base_strategy, num_tasks, num_latents, num_central_latents, num_quantiles
    ):
        raise NotImplementedError

    @property
    def inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    @property
    def lengthscale_prior_loc(self):
        return self.covar_module.base_kernel.lengthscale_prior.loc

    @property
    def lengthscale_prior_scale(self):
        return self.covar_module.base_kernel.lengthscale_prior.scale


class GPQR_Independent(BaseGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_central_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        num_latents = sum(num_quantiles)
        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            num_central_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape=batch_shape,
        )

    @staticmethod
    def construct_variational_strategy(
        base_strategy, num_tasks, num_latents, num_central_latents, num_quantiles
    ):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
        )


class GPQR_LMC(BaseGP):
    @staticmethod
    def construct_variational_strategy(
        base_strategy, num_tasks, num_latents, num_central_latents, num_quantiles
    ):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
        )


class GPQR_CenterGapLMC(BaseGP):
    @staticmethod
    def construct_variational_strategy(
        base_strategy, num_tasks, num_latents, num_central_latents, num_quantiles
    ):
        return CenterGapLMCVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
            num_central_latents=num_central_latents,
            num_quantiles=num_quantiles,
        )
