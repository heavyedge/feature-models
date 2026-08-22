import torch
from gpytorch.likelihoods import (
    MultitaskGaussianLikelihood as BaseMultitaskGaussianLikelihood,
)
from gpytorch.priors import LogNormalPrior
from gpytorch_qr.likelihoods import (
    MultioutputCenterGapQuantilesLikelihood as BaseCenterGapQuantilesLikelihood,
)

__all__ = [
    "MultitaskGaussianLikelihood",
    "MultioutputCenterGapQuantilesLikelihood",
]


def _make_noise_prior(loc, scale):
    if loc is None or scale is None:
        return None
    return LogNormalPrior(loc, scale)


class MultitaskGaussianLikelihood(BaseMultitaskGaussianLikelihood):
    def __init__(
        self,
        *args,
        noise_prior_loc=None,
        noise_prior_scale=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_prior = _make_noise_prior(noise_prior_loc, noise_prior_scale)
        kwargs.update(noise_prior=noise_prior)
        super().__init__(*args, **kwargs)
        self.noise_prior_loc = None if noise_prior is None else noise_prior.loc
        self.noise_prior_scale = None if noise_prior is None else noise_prior.scale
        self.batch_shape = batch_shape


class MultioutputCenterGapQuantilesLikelihood(BaseCenterGapQuantilesLikelihood):
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
