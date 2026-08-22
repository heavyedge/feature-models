import torch

from scripts.v1.model.gpqr import GPQR_Independent
from scripts.v1.model.gpr import GPR
from scripts.v1.model.likelihoods import GaussianLikelihood
from scripts.v1.model.prior import PriorMean


def test_prior_mean_uses_output_batch_axis():
    model = PriorMean(batch_shape=torch.Size([2]))
    X = torch.rand(2, 5, 3) + 0.5
    assert model(X).shape == (2, 3, 5)


def test_gpr_parameters_are_independent_under_a_shared_prior():
    inducing_points = torch.rand(2, 3, 4, 3)
    model = GPR(
        inducing_points,
        batch_shape=torch.Size([2, 3]),
        lengthscale_prior_loc=-1.0,
        lengthscale_prior_scale=0.5,
    )
    likelihood = GaussianLikelihood(
        batch_shape=torch.Size([2, 3]),
        noise_prior_loc=-4.0,
        noise_prior_scale=0.5,
    )

    assert model.covar_module.base_kernel.raw_lengthscale.shape == (2, 3, 1, 3)
    assert likelihood.raw_noise.shape == (2, 3, 1)
    assert model.lengthscale_prior_loc.numel() == 1
    assert likelihood.noise_prior_loc.numel() == 1
    assert model(torch.rand(2, 3, 5, 3)).mean.shape == (2, 3, 5)


def test_gpqr_tensor_contract():
    model = GPQR_Independent(
        torch.rand(2, 3, 4, 3),
        num_quantiles=3,
        num_lower_quantiles=1,
        num_latents=3,
        batch_shape=torch.Size([2, 3]),
        lengthscale_prior_loc=-1.0,
        lengthscale_prior_scale=0.5,
    )
    posterior = model.joint_quantile_posterior(torch.rand(2, 3, 5, 3))
    assert posterior.rsample().shape == (2, 3, 5, 3)
    assert model.covar_module.base_kernel.raw_lengthscale.shape == (2, 3, 3, 1, 3)
    assert model.lengthscale_prior_loc.numel() == 1
