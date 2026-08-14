import gpytorch
import torch
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior

__all__ = [
    "GPR_H",
    "GPR_phi",
]


class BaseGP(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        ls_prior = LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=D,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior,
            ),
            batch_shape=batch_shape,
        )
        self.covar_module = kernel

        self.batch_shape = batch_shape

    @property
    def lengthscale_prior_loc(self):
        return self.covar_module.base_kernel.lengthscale_prior.loc

    @property
    def lengthscale_prior_scale(self):
        return self.covar_module.base_kernel.lengthscale_prior.scale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def quantiles(self, x, quantiles):
        """Estimate quantile levels of response variable.

        Parameters
        ----------
        x: torch.Tensor in shape (*B, N, D)
        quantiles: torch.Tensor in shape (Q,)

        Returns
        -------
        quantiles_x: torch.Tensor in shape (*B, N, Q)
        """
        pred = self.likelihood(self(x))
        mean = pred.mean  # (*B, N)
        std = pred.variance.sqrt()  # (*B, N)
        z = torch.distributions.Normal(0, 1).icdf(quantiles)  # (Q,)
        return mean[..., None] + std[..., None] * z  # (*B, N, Q)


class GPR_H(BaseGP):
    pass


class GPR_phi(BaseGP):
    pass
