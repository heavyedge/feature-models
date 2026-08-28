from unittest.mock import Mock

import torch
from linear_operator.operators import BlockInterleavedLinearOperator

from scripts.v1.model.gpqr import GPQR_LMC, GPQR_Independent
from scripts.v1.model.likelihoods import CenterGapQuantilesLikelihood
from scripts.v1.train.common import _validation_expected_log_prob


class CapturingLikelihood:
    def expected_log_prob(self, y, function_dist):
        self.function_dist = function_dist
        return torch.zeros_like(y)


def make_posterior(model_class):
    torch.manual_seed(42)
    model = model_class(
        torch.rand(2, 3, 4, 3),
        num_quantiles=3,
        num_lower_quantiles=1,
        num_latents=3,
        batch_shape=torch.Size([2, 3]),
        lengthscale_prior_loc=-1.0,
        lengthscale_prior_scale=0.5,
    )
    model.eval()
    return model(torch.rand(2, 3, 5, 3))


def test_validation_builds_independent_task_covariance_from_variance():
    posterior = make_posterior(GPQR_Independent)
    assert isinstance(posterior.lazy_covariance_matrix, BlockInterleavedLinearOperator)
    expected_variance = posterior.variance + 1e-4
    posterior.to_data_independent_dist = Mock(
        side_effect=AssertionError("unsafe covariance indexing was used")
    )
    likelihood = CapturingLikelihood()

    _validation_expected_log_prob(
        likelihood,
        torch.rand(2, 3, 5),
        posterior,
        use_data_independent_samples=True,
        jitter=1e-4,
    )

    covariance = likelihood.function_dist.covariance_matrix
    assert covariance.shape == (2, 3, 5, 3, 3)
    torch.testing.assert_close(covariance.diagonal(dim1=-2, dim2=-1), expected_variance)
    torch.testing.assert_close(
        covariance - torch.diag_embed(expected_variance),
        torch.zeros_like(covariance),
    )
    quantile_likelihood = CenterGapQuantilesLikelihood(
        torch.tensor([0.05, 0.5, 0.95]),
        central_quantile_idx=1,
        batch_shape=torch.Size([2, 3]),
    )
    with torch.no_grad():
        log_prob = quantile_likelihood.expected_log_prob(
            torch.rand(2, 3, 5), likelihood.function_dist
        )
    assert log_prob.shape == (2, 3, 5)
    assert torch.isfinite(log_prob).all()


def test_validation_preserves_correlated_task_covariance():
    posterior = make_posterior(GPQR_LMC)
    assert not isinstance(
        posterior.lazy_covariance_matrix, BlockInterleavedLinearOperator
    )
    expected = posterior.to_data_independent_dist(jitter_val=1e-4)
    original_conversion = posterior.to_data_independent_dist
    posterior.to_data_independent_dist = Mock(wraps=original_conversion)
    likelihood = CapturingLikelihood()

    _validation_expected_log_prob(
        likelihood,
        torch.rand(2, 3, 5),
        posterior,
        use_data_independent_samples=True,
        jitter=1e-4,
    )

    posterior.to_data_independent_dist.assert_called_once_with(jitter_val=1e-4)
    torch.testing.assert_close(
        likelihood.function_dist.covariance_matrix,
        expected.covariance_matrix,
    )
