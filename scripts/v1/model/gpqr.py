import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
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
    """Quantile GP whose output variables are independent GP batches."""

    output_names = ("H", "phi_1", "phi_3")

    def __init__(
        self,
        inducing_points,
        num_quantiles,
        central_quantile_idx,
        num_latents,
        lengthscale,
        batch_shape=torch.Size(),
        quantile_levels=None,
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
            num_quantiles,
            num_latents,
        )

        mean = ConstantMean(batch_shape=full_batch_shape)
        covar = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=full_batch_shape,
            ),
            batch_shape=full_batch_shape,
        )

        super().__init__(
            variational_strategy,
            mean,
            covar,
            [num_quantiles],
            [central_quantile_idx],
            quantile_levels=(None if quantile_levels is None else [quantile_levels]),
        )

        self.num_latents = num_latents
        self.batch_shape = torch.Size(batch_shape)
        self._freeze_lengthscale(lengthscale, D)

    def _freeze_lengthscale(self, lengthscale, dim):
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
                    "lengthscale must have shape "
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


class GPQR_Independent(BaseGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        central_quantile_idx,
        num_latents,
        lengthscale,
        batch_shape=torch.Size(),
        quantile_levels=None,
    ):
        super().__init__(
            inducing_points,
            num_quantiles,
            central_quantile_idx,
            num_quantiles,
            lengthscale,
            batch_shape,
            quantile_levels,
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
