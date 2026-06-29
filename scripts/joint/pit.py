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


def _interpolate_linear(q_values, q_levels, threshold):
    idx = np.array([np.searchsorted(row, threshold) for row in q_values])
    idx_clamped = np.clip(idx, 1, len(q_levels) - 1)

    rows = np.arange(len(q_values))
    x0 = q_values[rows, idx_clamped - 1]
    x1 = q_values[rows, idx_clamped]
    y0 = q_levels[idx_clamped - 1]
    y1 = q_levels[idx_clamped]
    probs = y0 + (threshold - x0) * (y1 - y0) / (x1 - x0)

    probs = np.where(idx == 0, 0.0, probs)
    probs = np.where(idx == len(q_levels), 1.0, probs)
    return np.clip(probs, 0.0, 1.0)


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
    N, Q = q_values.shape
    idx = np.array([np.searchsorted(q_values[i], thresholds[i]) for i in range(N)])
    idx_clamped = np.clip(idx, 1, Q - 1)

    rows = np.arange(N)
    x0 = q_values[rows, idx_clamped - 1]
    x1 = q_values[rows, idx_clamped]
    y0 = q_levels[idx_clamped - 1]
    y1 = q_levels[idx_clamped]
    probs = y0 + (thresholds - x0) * (y1 - y0) / (x1 - x0)

    probs = np.where(idx == 0, 0.0, probs)
    probs = np.where(idx == Q, 1.0, probs)
    return np.clip(probs, 0.0, 1.0)
