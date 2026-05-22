import gpytorch
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
from gpytorch_qr.means import CenterGapMean
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP, QuantileGP
from gpytorch_qr.variational import CGBlkdiagLmcVariationalStrategy

__all__ = [
    "Unscaler",
    "PriorMean_H",
    "GPR_H",
    "GPR_phi",
    "CgLmcMtgpqr_H",
    "CgLmcMtgpqr_H_ConstantMean",
    "CgLmcMtgpqr_phi",
    "CgIndependentMtgpqr_H",
    "CgIndependentMtgpqr_H_ConstantMean",
    "CgIndependentMtgpqr_phi",
    "DirectLmcMtgpqr_H",
    "DirectLmcMtgpqr_H_ConstantMean",
    "DirectLmcMtgpqr_phi",
    "DirectIndependentMtgpqr_H",
    "DirectIndependentMtgpqr_H_ConstantMean",
    "DirectIndependentMtgpqr_phi",
    "save_model",
    "load_model",
]


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling.

    Parameters
    ----------
    X_scale, X_min: torch.Tensor in shape (*B, D)
    """

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        # x: (*B, 1, N, D)
        # BELOW IS CORRECT FOR MinMaxScaler (different from StandardScaler)
        X_min = self.X_min[..., None, None, :]
        X_scale = self.X_scale[..., None, None, :]
        x_unscaled = (x - X_min) / X_scale
        return x_unscaled.view_as(x)


class PriorMean_H(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self, offset=False, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape
        if offset:
            self.register_parameter(
                "offset",
                torch.nn.Parameter(torch.zeros(batch_shape)),
            )
        else:
            self.register_buffer(
                "offset",
                torch.zeros(batch_shape),
            )

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a, b, c = 0.22, -0.43, 0.77  # From GPR prior
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model + self.offset[..., None]  # (*B, N)


# Gaussian proces regression


class _PriorMean_H_GPR(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self, offset=False, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape
        if offset:
            self.register_parameter(
                "offset",
                torch.nn.Parameter(torch.zeros(batch_shape)),
            )
        else:
            self.register_buffer(
                "offset",
                torch.zeros(batch_shape),
            )

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a, b, c = 0.22, -0.43, 0.77  # From GPR prior
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model + self.offset[..., None, None]


class GPR_H(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        self.mean_module = torch.nn.Sequential(
            unscaler,
            _PriorMean_H_GPR(offset=False, batch_shape=batch_shape),
        )
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x.unsqueeze(-3)).squeeze(-2)
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


class GPR_phi(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        self.mean_module = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        self.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = init_ls

    def forward(self, x):
        mean_x = self.mean_module(x.unsqueeze(-3)).squeeze(-2)
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


# Center-gap LMC MTGPQR


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
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
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


class CgLmcMtgpqr_H_ConstantMean(CenterGapQuantileGP):
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
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgLmcMtgpqr_phi(CenterGapQuantileGP):
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
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
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


# Center-gap Independent MTGPQR


class CgIndependentMtgpqr_H(CenterGapQuantileGP):
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


class CgIndependentMtgpqr_H_ConstantMean(CenterGapQuantileGP):
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
                ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
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


# Direct LMC MTGPQR


class DirectLmcMtgpqr_H(DirectQuantileGP):
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
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            PriorMean_H(offset=True, batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectLmcMtgpqr_H_ConstantMean(DirectQuantileGP):
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
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectLmcMtgpqr_phi(DirectQuantileGP):
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
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
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

        super().__init__(variational_strategy, mean, covar, -1)


# Direct Independent MTGPQR


class DirectIndependentMtgpqr_H(DirectQuantileGP):
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

        mean = torch.nn.Sequential(
            unscaler,
            PriorMean_H(offset=True, batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectIndependentMtgpqr_H_ConstantMean(DirectQuantileGP):
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

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectIndependentMtgpqr_phi(DirectQuantileGP):
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

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
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

        super().__init__(variational_strategy, mean, covar, -1)


def save_model(
    train_x,
    train_y,
    model,
    likelihood,
    scaler,
    inducing_points,
    quantiles,
    num_lower_quantiles,
    num_latents,
    num_lower_latents,
    path,
):
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "scaler": scaler,
            "inducing_points": inducing_points,
            "quantiles": quantiles,
            "num_lower_quantiles": num_lower_quantiles,
            "num_latents": num_latents,
            "num_lower_latents": num_lower_latents,
        },
        path,
    )


def load_model(model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if issubclass(model_class, ExactGP):
        likelihood = GaussianLikelihood()
    elif issubclass(model_class, CenterGapQuantileGP):
        likelihood = MultitaskCenterGapQuantileGPLikelihood(
            checkpoint["quantiles"],
            checkpoint["num_lower_quantiles"],
        )
    elif issubclass(model_class, DirectQuantileGP):
        likelihood = MultitaskQuantileGPLikelihood(
            checkpoint["quantiles"],
        )
    else:
        raise ValueError("Unsupported model class.")
    if issubclass(model_class, ExactGP):
        model = model_class(
            train_x=checkpoint["train_x"],
            train_y=checkpoint["train_y"],
            likelihood=likelihood,
        )
    if issubclass(model_class, QuantileGP):
        model = model_class(
            inducing_points=checkpoint["inducing_points"],
            num_quantiles=len(checkpoint["quantiles"]),
            num_lower_quantiles=checkpoint["num_lower_quantiles"],
            num_latents=checkpoint["num_latents"],
            num_lower_latents=checkpoint["num_lower_latents"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    if device is not None:
        model.to(device)
        likelihood.to(device)
    scaler = checkpoint["scaler"]
    return model, likelihood, scaler
