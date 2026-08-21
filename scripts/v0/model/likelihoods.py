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


def _make_noise_prior(loc, scale):
    if loc is None or scale is None:
        return None
    return LogNormalPrior(loc, scale)


class GaussianLikelihood(BaseGaussianLikelihood):
    def __init__(self, *args, noise_prior_loc=None, noise_prior_scale=None, **kwargs):
        kwargs.update(noise_prior=_make_noise_prior(noise_prior_loc, noise_prior_scale))
        super().__init__(*args, **kwargs)

    @property
    def noise_prior_loc(self):
        prior = getattr(self.noise_covar, "noise_prior", None)
        return None if prior is None else prior.loc

    @property
    def noise_prior_scale(self):
        prior = getattr(self.noise_covar, "noise_prior", None)
        return None if prior is None else prior.scale

    @property
    def batch_shape(self):
        return self.noise_covar.raw_noise.shape[:-1]


class CenterGapQuantilesLikelihood(BaseCenterGapQuantilesLikelihood):
    def __init__(
        self,
        quantile_levels,
        central_quantile_idx,
        *args,
        noise_prior_loc=None,
        noise_prior_scale=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_prior = _make_noise_prior(noise_prior_loc, noise_prior_scale)
        super().__init__(
            quantile_levels,
            central_quantile_idx,
            *args,
            noise_prior=noise_prior,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.quantile_levels = quantile_levels
        self.central_quantile_idx = central_quantile_idx
        self.noise_prior_loc = None if noise_prior is None else noise_prior.loc
        self.noise_prior_scale = None if noise_prior is None else noise_prior.scale
        self.batch_shape = batch_shape


class DirectQuantilesLikelihood(BaseDirectQuantilesLikelihood):
    def __init__(
        self,
        quantile_levels,
        *args,
        noise_prior_loc=None,
        noise_prior_scale=None,
        batch_shape=torch.Size(),
        central_quantile_idx=0,  # dummy argument
        **kwargs,
    ):
        noise_prior = _make_noise_prior(noise_prior_loc, noise_prior_scale)
        super().__init__(
            quantile_levels,
            *args,
            noise_prior=noise_prior,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.quantile_levels = quantile_levels
        self.central_quantile_idx = central_quantile_idx
        self.noise_prior_loc = None if noise_prior is None else noise_prior.loc
        self.noise_prior_scale = None if noise_prior is None else noise_prior.scale
        self.batch_shape = batch_shape
