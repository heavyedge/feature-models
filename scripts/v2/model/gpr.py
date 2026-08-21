import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)

__all__ = [
    "GPR_Independent",
    "GPR_LMC",
]


class BaseGP(ApproximateGP):

    output_names = ("H", "phi_1", "phi_3")
    num_tasks = len(output_names)

    def __init__(
        self,
        inducing_points,
        num_latents,
        lengthscale_prior_loc=None,
        lengthscale_prior_scale=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
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
                learn_inducing_locations=False,
            ),
            self.num_tasks,
            num_latents,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        ls_prior = None
        if lengthscale_prior_loc is not None and lengthscale_prior_scale is not None:
            ls_prior = LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale)
        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior,
            ),
            batch_shape=batch_shape,
        )

        self.num_latents = num_latents
        self.batch_shape = batch_shape

    @staticmethod
    def construct_variational_strategy(base_strategy, num_tasks, num_latents):
        raise NotImplementedError

    @property
    def inducing_points(self):
        # Multitask strategies wrap the inducing-point variational strategy.
        return self.variational_strategy.base_variational_strategy.inducing_points

    @property
    def lengthscale_prior_loc(self):
        prior = getattr(self.covar_module.base_kernel, "lengthscale_prior", None)
        return None if prior is None else prior.loc

    @property
    def lengthscale_prior_scale(self):
        prior = getattr(self.covar_module.base_kernel, "lengthscale_prior", None)
        return None if prior is None else prior.scale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPR_Independent(BaseGP):
    def __init__(
        self,
        inducing_points,
        num_latents,
        lengthscale_prior_loc=None,
        lengthscale_prior_scale=None,
        batch_shape=torch.Size(),
    ):
        num_latents = self.num_tasks
        super().__init__(
            inducing_points,
            num_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_tasks, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
        )


class GPR_LMC(BaseGP):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_tasks, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
        )
