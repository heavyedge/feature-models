import gpytorch
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

__all__ = [
    "GPR_H_2",
    "GPR_b_2",
    "GPR_phi_2",
]


class GPR_H_2(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

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


class GPR_b_2(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=batch_shape)

        lower = torch.tensor([0.01, 0.01, 0.01] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4, 1e4, 1e4] + [1e4 for _ in range(D - 3)])
        init_ls = torch.tensor([0.5, 0.5, 0.5] + [0.5 for _ in range(D - 3)])
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        kernel.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            kernel.base_kernel.lengthscale = init_ls
        self.covar_module = kernel

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


class GPR_phi_2(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=batch_shape)

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        kernel.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            kernel.base_kernel.lengthscale = init_ls
        self.covar_module = kernel

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
