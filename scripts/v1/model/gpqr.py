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

from .quantile import (
    DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    LowerBoundedCenterGapToQuantileTransform,
    quantile_slope_offsets,
)

__all__ = [
    "GPQR_Independent",
    "GPQR_LMC",
    "GPQR_CenterGapLMC",
]


class BaseGP(CenterGapQuantileGP):
    """Quantile GP whose output variables are independent GP batches."""

    output_names = ("H", "phi_1", "phi_3")

    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=None,
        lengthscale_prior_scale=None,
        fixed_lengthscale=None,
        batch_shape=torch.Size(),
        quantile_levels=None,
        quantile_slope_lower_bound=DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    ):
        N, D = inducing_points.shape[-2:]
        if fixed_lengthscale is not None and (
            lengthscale_prior_loc is not None or lengthscale_prior_scale is not None
        ):
            raise ValueError(
                "fixed_lengthscale cannot be combined with a lengthscale prior"
            )
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
            num_quantiles,
            num_latents,
        )

        mean = ConstantMean(batch_shape=full_batch_shape)
        lengthscale_prior = None
        if lengthscale_prior_loc is not None and lengthscale_prior_scale is not None:
            lengthscale_prior = LogNormalPrior(
                lengthscale_prior_loc, lengthscale_prior_scale
            )
        covar = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=full_batch_shape,
                lengthscale_prior=lengthscale_prior,
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

        self.num_latents = num_latents
        self.batch_shape = torch.Size(batch_shape)
        self.quantile_slope_lower_bound = float(quantile_slope_lower_bound)
        self.register_buffer("quantile_levels", None, persistent=False)
        if quantile_levels is not None:
            self.set_quantile_levels(quantile_levels)
        self.lengthscale_is_fixed = fixed_lengthscale is not None
        if fixed_lengthscale is not None:
            self._set_fixed_lengthscale(fixed_lengthscale, D)

    def set_quantile_levels(self, quantile_levels):
        transform = LowerBoundedCenterGapToQuantileTransform(
            quantile_levels,
            self.num_lower_quantiles[0],
            self.quantile_slope_lower_bound,
        )
        self.quantile_levels = transform.quantile_levels.detach().clone()

    def _quantile_transform(self):
        if self.quantile_levels is None:
            raise RuntimeError("quantile_levels must be set before prediction")
        return LowerBoundedCenterGapToQuantileTransform(
            self.quantile_levels,
            self.num_lower_quantiles[0],
            self.quantile_slope_lower_bound,
        )

    def joint_quantile_posterior(self, x):
        return torch.distributions.TransformedDistribution(
            self(x), self._quantile_transform()
        )

    def mean_quantiles_delta(self, x):
        quantiles = super().mean_quantiles_delta(x)
        return quantiles + quantile_slope_offsets(
            self.quantile_levels,
            self.num_lower_quantiles[0],
            self.quantile_slope_lower_bound,
            like=quantiles,
        )

    def _set_fixed_lengthscale(self, lengthscale, dim):
        """Copy a GPR ARD lengthscale across GPQR latent processes and freeze it."""
        kernel = self.covar_module.base_kernel
        target = kernel.lengthscale
        lengthscale = torch.as_tensor(
            lengthscale,
            dtype=target.dtype,
            device=target.device,
        )

        if lengthscale.shape == target.shape:
            expanded = lengthscale
        else:
            if lengthscale.shape[-2:] == (1, dim):
                lengthscale = lengthscale.squeeze(-2)
            shared_shape = self.batch_shape[-1:] + torch.Size((dim,))
            batched_shape = self.batch_shape + torch.Size((dim,))
            if lengthscale.shape == batched_shape:
                expanded = lengthscale.reshape(
                    *self.batch_shape,
                    1,
                    1,
                    dim,
                ).expand_as(target)
            elif lengthscale.shape == shared_shape:
                external_dims = len(self.batch_shape) - 1
                expanded = lengthscale.reshape(
                    *((1,) * external_dims),
                    self.batch_shape[-1],
                    1,
                    1,
                    dim,
                ).expand_as(target)
            else:
                raise ValueError(
                    "fixed_lengthscale must have shape "
                    f"{tuple(shared_shape)}, {tuple(batched_shape)}, "
                    f"or {tuple(target.shape)}; got {tuple(lengthscale.shape)}"
                )

        kernel.lengthscale = expanded
        kernel.raw_lengthscale.requires_grad_(False)

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        raise NotImplementedError

    @property
    def inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    @property
    def lengthscale_prior_loc(self):
        prior = getattr(self.covar_module.base_kernel, "lengthscale_prior", None)
        return None if prior is None else prior.loc

    @property
    def lengthscale_prior_scale(self):
        prior = getattr(self.covar_module.base_kernel, "lengthscale_prior", None)
        return None if prior is None else prior.scale


class GPQR_Independent(BaseGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=None,
        lengthscale_prior_scale=None,
        fixed_lengthscale=None,
        batch_shape=torch.Size(),
        quantile_levels=None,
        quantile_slope_lower_bound=DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    ):
        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_quantiles,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            fixed_lengthscale,
            batch_shape,
            quantile_levels,
            quantile_slope_lower_bound,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
        )


class GPQR_LMC(BaseGP):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


class GPQR_CenterGapLMC(BaseGP):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return CenterGapLMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )
