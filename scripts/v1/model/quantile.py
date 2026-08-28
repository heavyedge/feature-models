import math

import torch
from gpytorch_qr.utils import CenterGapToQuantileTransform

__all__ = [
    "DEFAULT_QUANTILE_SLOPE_LOWER_BOUND",
    "LowerBoundedCenterGapToQuantileTransform",
    "quantile_slope_offsets",
]


DEFAULT_QUANTILE_SLOPE_LOWER_BOUND = 1e-4


def _validate_quantile_parameters(
    quantile_levels, central_quantile_idx, quantile_slope_lower_bound
):
    levels = torch.as_tensor(quantile_levels)
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError("quantile_levels must be a 1D tensor with at least two values")
    if not torch.isfinite(levels).all() or not (levels.diff() > 0).all():
        raise ValueError("quantile_levels must be finite and strictly increasing")

    central_quantile_idx = int(central_quantile_idx)
    if not 0 <= central_quantile_idx < len(levels):
        raise ValueError("central_quantile_idx is outside quantile_levels")

    lower_bound = float(quantile_slope_lower_bound)
    if not math.isfinite(lower_bound) or lower_bound <= 0:
        raise ValueError("quantile_slope_lower_bound must be finite and positive")
    return levels, central_quantile_idx, lower_bound


def quantile_slope_offsets(
    quantile_levels,
    central_quantile_idx,
    quantile_slope_lower_bound=DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    *,
    like=None,
):
    """Return offsets that add a lower bound to every quantile secant slope."""
    levels, central_quantile_idx, lower_bound = _validate_quantile_parameters(
        quantile_levels, central_quantile_idx, quantile_slope_lower_bound
    )
    if like is not None:
        levels = levels.to(dtype=like.dtype, device=like.device)
    return lower_bound * (levels - levels[central_quantile_idx])


class LowerBoundedCenterGapToQuantileTransform(CenterGapToQuantileTransform):
    """Center-gap transform with a positive lower bound on quantile slope."""

    def __init__(
        self,
        quantile_levels,
        central_quantile_idx,
        quantile_slope_lower_bound=DEFAULT_QUANTILE_SLOPE_LOWER_BOUND,
    ):
        levels, central_quantile_idx, lower_bound = _validate_quantile_parameters(
            quantile_levels, central_quantile_idx, quantile_slope_lower_bound
        )
        super().__init__([len(levels)], [central_quantile_idx])
        self.quantile_levels = levels
        self.central_quantile_idx = central_quantile_idx
        self.quantile_slope_lower_bound = lower_bound

    def _slope_offsets(self, value):
        return quantile_slope_offsets(
            self.quantile_levels,
            self.central_quantile_idx,
            self.quantile_slope_lower_bound,
            like=value,
        )

    def _call(self, x):
        return super()._call(x) + self._slope_offsets(x)

    def _inverse(self, y):
        return super()._inverse(y - self._slope_offsets(y))
