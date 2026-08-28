import pytest
import torch

from scripts.v1.model.likelihoods import CenterGapQuantilesLikelihood
from scripts.v1.model.quantile import (
    LowerBoundedCenterGapToQuantileTransform,
    quantile_slope_offsets,
)

QUANTILE_LEVELS = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])


def test_center_gap_transform_enforces_quantile_slope_lower_bound():
    lower_bound = 1e-4
    transform = LowerBoundedCenterGapToQuantileTransform(
        QUANTILE_LEVELS,
        central_quantile_idx=2,
        quantile_slope_lower_bound=lower_bound,
    )
    # Without the added offsets, these tiny softplus gaps round back to the
    # float32 center and produce tied quantiles.
    center_and_gaps = torch.tensor([[1.0, -100.0, -100.0, -100.0, -100.0]])

    quantiles = transform(center_and_gaps)
    slopes = quantiles.diff(dim=-1) / QUANTILE_LEVELS.diff()

    assert torch.all(slopes >= lower_bound * 0.99)
    assert torch.all(quantiles.diff(dim=-1) > 0)


def test_quantile_slope_offsets_leave_central_quantile_unchanged():
    offsets = quantile_slope_offsets(
        QUANTILE_LEVELS,
        central_quantile_idx=2,
        quantile_slope_lower_bound=1e-4,
    )

    assert offsets[2] == 0
    assert torch.allclose(offsets.diff(), 1e-4 * QUANTILE_LEVELS.diff())


def test_likelihood_uses_the_same_lower_bounded_transform():
    lower_bound = 1e-4
    likelihood = CenterGapQuantilesLikelihood(
        QUANTILE_LEVELS,
        central_quantile_idx=2,
        quantile_slope_lower_bound=lower_bound,
    )
    center_and_gaps = torch.tensor([[[1.0, -100.0, -100.0, -100.0, -100.0]]])

    quantiles = likelihood.forward(center_and_gaps).base_dist.loc
    slopes = quantiles.diff(dim=-1) / QUANTILE_LEVELS.diff()

    assert torch.all(slopes >= lower_bound * 0.99)
    assert torch.all(quantiles.diff(dim=-1) > 0)


@pytest.mark.parametrize("lower_bound", [0, -1, float("inf"), float("nan")])
def test_quantile_slope_lower_bound_must_be_finite_and_positive(lower_bound):
    with pytest.raises(ValueError, match="finite and positive"):
        quantile_slope_offsets(QUANTILE_LEVELS, 2, lower_bound)
