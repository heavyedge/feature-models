import numpy as np

__all__ = [
    "quantile_interpolation",
    "quantile_pit",
]


def quantile_interpolation(q_values, q_levels, threshold):
    """Estimate P(Y <= threshold) from predicted quantiles.

    Parameters
    ----------
    q_values : (N, Q) array
        Predicted quantile values, sorted along axis=1 (no crossing).
    q_levels : (Q,) array
        Quantile levels (taus), sorted in ascending order.
    threshold : float
        The threshold value.
    """
    return _interpolate_linear(q_values, q_levels, threshold)


def _interpolate_linear(q_values, q_levels, thresholds):
    q_values = np.asarray(q_values)
    q_levels = np.asarray(q_levels)
    thresholds = np.asarray(thresholds)

    N, Q = q_values.shape
    if thresholds.ndim == 0:
        thresholds = np.full(N, thresholds)
    else:
        thresholds = thresholds.reshape(-1)
    if thresholds.shape[0] != N:
        raise ValueError(
            f"thresholds must be scalar or have length {N}; got {thresholds.shape[0]}"
        )

    idx = np.array(
        [np.searchsorted(q_values[i], thresholds[i], side="right") for i in range(N)]
    )
    idx_clamped = np.clip(idx, 1, Q - 1)

    rows = np.arange(N)
    x0 = q_values[rows, idx_clamped - 1]
    x1 = q_values[rows, idx_clamped]
    y0 = q_levels[idx_clamped - 1]
    y1 = q_levels[idx_clamped]

    slope = np.divide(
        y1 - y0,
        x1 - x0,
        out=np.zeros_like(y1, dtype=float),
        where=x1 != x0,
    )
    probs = y0 + (thresholds - x0) * slope

    probs = np.where(idx == 0, q_levels[0], probs)
    probs = np.where(idx == Q, q_levels[-1], probs)
    return np.clip(probs, q_levels[0], q_levels[-1])


def quantile_pit(q_values, q_levels, thresholds):
    """Compute PIT values P(Y <= y_i | x_i) with per-sample thresholds.

    Parameters
    ----------
    q_values : (N, Q) array
        Predicted quantile values, sorted along axis=1 (no crossing).
    q_levels : (Q,) array
        Quantile levels (taus), sorted in ascending order.
    thresholds : (N,) array
        Per-sample threshold (actual observed values).

    Returns
    -------
    (N,) array
        Estimated CDF value for each sample.
    """
    return _interpolate_linear(q_values, q_levels, thresholds)
