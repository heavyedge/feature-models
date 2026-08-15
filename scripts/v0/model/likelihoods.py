import torch
from gpytorch.likelihoods import GaussianLikelihood as BaseGaussianLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch_qr.likelihoods import (
    CenterGapQuantilesLikelihood as BaseCenterGapQuantilesLikelihood,
)
from gpytorch_qr.likelihoods import (
    DirectQuantilesLikelihood as BaseDirectQuantilesLikelihood,
)

__all__ = [
    "GaussianLikelihood",
    "CenterGapQuantilesLikelihood",
    "DirectQuantilesLikelihood",
]


class GaussianLikelihood(BaseGaussianLikelihood):
    def __init__(self, *args, noise_prior_loc=0.0, noise_prior_scale=1.0, **kwargs):
        kwargs.update(
            dict(noise_prior=LogNormalPrior(noise_prior_loc, noise_prior_scale))
        )
        super().__init__(*args, **kwargs)

    @property
    def noise_prior_loc(self):
        return self.noise_covar.noise_prior.loc

    @property
    def noise_prior_scale(self):
        return self.noise_covar.noise_prior.scale

    @property
    def batch_shape(self):
        return self.noise_covar.raw_noise.shape[:-1]


class CenterGapQuantilesLikelihood(BaseCenterGapQuantilesLikelihood):
    def __init__(
        self,
        quantile_levels,
        central_quantile_idx,
        *args,
        noise_prior_loc=0.0,
        noise_prior_scale=1.0,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        super().__init__(
            quantile_levels,
            central_quantile_idx,
            *args,
            noise_prior=LogNormalPrior(noise_prior_loc, noise_prior_scale),
            batch_shape=batch_shape,
            **kwargs,
        )
        self.quantile_levels = quantile_levels
        self.central_quantile_idx = central_quantile_idx
        self.noise_prior_loc = noise_prior_loc
        self.noise_prior_scale = noise_prior_scale
        self.batch_shape = batch_shape


class DirectQuantilesLikelihood(BaseDirectQuantilesLikelihood):
    def __init__(
        self,
        quantile_levels,
        *args,
        noise_prior_loc=0.0,
        noise_prior_scale=1.0,
        batch_shape=torch.Size(),
        central_quantile_idx=0,  # dummy argument
        **kwargs,
    ):
        super().__init__(
            quantile_levels,
            *args,
            noise_prior=LogNormalPrior(noise_prior_loc, noise_prior_scale),
            batch_shape=batch_shape,
            **kwargs,
        )
        self.quantile_levels = quantile_levels
        self.central_quantile_idx = central_quantile_idx
        self.noise_prior_loc = noise_prior_loc
        self.noise_prior_scale = noise_prior_scale
        self.batch_shape = batch_shape
