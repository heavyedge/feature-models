from gpytorch.likelihoods import GaussianLikelihood as BaseGaussianLikelihood
from gpytorch.priors import LogNormalPrior

__all__ = [
    "GaussianLikelihood",
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
