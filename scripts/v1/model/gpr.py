import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

__all__ = [
    "GPR",
]


class BaseGP(ApproximateGP):
    output_names = ("H", "phi_1", "phi_3")

    def __init__(
        self,
        inducing_points,
        lengthscale_prior_loc=None,
        lengthscale_prior_scale=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
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

        self.batch_shape = torch.Size(batch_shape)

    @property
    def inducing_points(self):
        return self.variational_strategy.inducing_points

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


class GPR(BaseGP):
    pass
