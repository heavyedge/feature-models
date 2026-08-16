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
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP
from gpytorch_qr.variational import CenterGapLMCVariationalStrategy

__all__ = [
    "CenterGapMTGPQR_Independent_H",
    "CenterGapMTGPQR_Independent_phi",
    "CenterGapMTGPQR_LMC_H",
    "CenterGapMTGPQR_LMC_phi",
    "CenterGapMTGPQR_CenterGapLMC_H",
    "CenterGapMTGPQR_CenterGapLMC_phi",
    "CenterGapMTGPQR_H",
    "CenterGapMTGPQR_phi",
    "DirectMTGPQR_Independent_H",
    "DirectMTGPQR_Independent_phi",
    "DirectMTGPQR_LMC_H",
    "DirectMTGPQR_LMC_phi",
]


class _CGMTGPQR_Base(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
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
            num_quantiles,
            num_latents,
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
            variational_strategy,
            mean,
            covar,
            [num_quantiles],
            [num_lower_quantiles],
        )

        self.num_latents = num_latents
        self.batch_shape = batch_shape

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
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


class CenterGapMTGPQR_Independent_H(_CGMTGPQR_Base):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
        )


class CenterGapMTGPQR_Independent_phi(_CGMTGPQR_Base):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
        )


class CenterGapMTGPQR_LMC_H(_CGMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


class CenterGapMTGPQR_LMC_phi(_CGMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


class CenterGapMTGPQR_CenterGapLMC_H(_CGMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return CenterGapLMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


class CenterGapMTGPQR_CenterGapLMC_phi(_CGMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return CenterGapLMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


CenterGapMTGPQR_H = CenterGapMTGPQR_CenterGapLMC_H


CenterGapMTGPQR_phi = CenterGapMTGPQR_CenterGapLMC_phi


class _DirectMTGPQR_Base(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
        num_lower_quantiles=0,  # dummy argument
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
        ls_prior = LogNormalPrior(lengthscale_prior_loc, lengthscale_prior_scale)
        covar = ScaleKernel(
            RBFKernel(
                ard_num_dims=D,
                batch_shape=full_batch_shape,
                lengthscale_prior=ls_prior,
            ),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar)

        self.num_latents = num_latents
        self.batch_shape = batch_shape
        self.num_quantiles = [num_quantiles]  # dummy argument
        self.num_lower_quantiles = [num_lower_quantiles]  # dummy argument

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
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


class DirectMTGPQR_Independent_H(_DirectMTGPQR_Base):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
        num_lower_quantiles=0,  # dummy argument
    ):
        num_latents = num_quantiles
        super().__init__(
            inducing_points,
            num_quantiles,
            num_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape,
            num_lower_quantiles,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
        )


class DirectMTGPQR_Independent_phi(_DirectMTGPQR_Base):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_latents,
        lengthscale_prior_loc=0.0,
        lengthscale_prior_scale=1.0,
        batch_shape=torch.Size(),
        num_lower_quantiles=0,  # dummy argument
    ):
        num_latents = num_quantiles
        super().__init__(
            inducing_points,
            num_quantiles,
            num_latents,
            lengthscale_prior_loc,
            lengthscale_prior_scale,
            batch_shape,
            num_lower_quantiles,
        )

    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
        )


class DirectMTGPQR_LMC_H(_DirectMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )


class DirectMTGPQR_LMC_phi(_DirectMTGPQR_Base):
    @staticmethod
    def construct_variational_strategy(base_strategy, num_quantiles, num_latents):
        return LMCVariationalStrategy(
            base_strategy,
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )
