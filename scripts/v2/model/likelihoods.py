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


class MultitaskGaussianLikelihood(BaseMultitaskGaussianLikelihood):
    def __init__(
        self,
        *args,
        noise_prior_loc=0.0,
        noise_prior_scale=1.0,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        kwargs.update(
            dict(noise_prior=LogNormalPrior(noise_prior_loc, noise_prior_scale))
        )
        super().__init__(*args, **kwargs)
        self.noise_prior_loc = noise_prior_loc
        self.noise_prior_scale = noise_prior_scale
        self.batch_shape = batch_shape


class MultioutputCenterGapQuantilesLikelihood(BaseCenterGapQuantilesLikelihood):
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
