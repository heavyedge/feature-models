import torch
from gpytorch.likelihoods import GaussianLikelihood as BaseGaussianLikelihood
from gpytorch_qr.likelihoods import (
    CenterGapQuantilesLikelihood as BaseCenterGapQuantilesLikelihood,
)

__all__ = [
    "GaussianLikelihood",
    "CenterGapQuantilesLikelihood",
]


class GaussianLikelihood(BaseGaussianLikelihood):
    @property
    def batch_shape(self):
        return self.noise_covar.raw_noise.shape[:-1]


class CenterGapQuantilesLikelihood(BaseCenterGapQuantilesLikelihood):
    def __init__(
        self,
        quantile_levels,
        central_quantile_idx,
        *args,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        super().__init__(
            quantile_levels,
            central_quantile_idx,
            *args,
            batch_shape=batch_shape,
            **kwargs,
        )
        self.batch_shape = torch.Size(batch_shape)
