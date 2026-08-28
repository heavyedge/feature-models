import torch
from gpytorch_qr.settings import (
    enforce_strict_quantile_order,
    quantile_gap_lower_bound,
)
from gpytorch_qr.utils import CenterGapToQuantileTransform

from scripts.v1.model.likelihoods import CenterGapQuantilesLikelihood

QUANTILE_LEVELS = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])


def test_likelihood_uses_gpytorch_qr_quantile_gap_lower_bound():
    lower_bound = 1e-4
    likelihood = CenterGapQuantilesLikelihood(
        QUANTILE_LEVELS,
        central_quantile_idx=2,
    )
    center_and_gaps = torch.tensor([[[1.0, -100.0, -100.0, -100.0, -100.0]]])

    with quantile_gap_lower_bound(lower_bound):
        quantiles = likelihood.forward(center_and_gaps).base_dist.loc
    slopes = quantiles.diff(dim=-1) / QUANTILE_LEVELS.diff()

    assert torch.all(slopes >= lower_bound * 0.99)
    assert torch.all(quantiles.diff(dim=-1) > 0)
    assert quantile_gap_lower_bound.value() == 0


def test_strict_quantile_order_preserves_prediction_dtype():
    transform = CenterGapToQuantileTransform([5], [2])
    center_and_gaps = torch.tensor(
        [[[1e8, -100.0, -100.0, -100.0, -100.0]]],
        dtype=torch.float32,
    )

    with enforce_strict_quantile_order(True):
        quantiles = transform(center_and_gaps)

    assert quantiles.dtype == torch.float32
    assert torch.all(quantiles.diff(dim=-1) > 0)
    assert enforce_strict_quantile_order.value() is False
