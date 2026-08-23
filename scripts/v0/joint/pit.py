import numpy as np
from backend import cuda_device

__all__ = [
    "quantile_interpolation",
    "quantile_pit",
]


def quantile_interpolation(
    q_values, q_levels, threshold, device="auto", chunk_size=262144
):
    """Estimate P(Y <= threshold) from predicted quantiles.

    Parameters
    ----------
    q_values : (N, Q) array
        Predicted quantile values, sorted along axis=1 (no crossing).
    q_levels : (Q,) array
        Quantile levels (taus), sorted in ascending order.
    threshold : float
        The threshold value.
    device : str
        ``"auto"`` selects CUDA when available; ``"cpu"`` forces NumPy.
    chunk_size : int
        Maximum rows transferred to CUDA at once.
    """
    return _interpolate_linear(q_values, q_levels, threshold, device, chunk_size)


def _interpolate_linear(q_values, q_levels, thresholds, device, chunk_size):
    q_values = np.asarray(q_values)
    q_levels = np.asarray(q_levels)
    thresholds = np.asarray(thresholds)

    if q_values.ndim != 2:
        raise ValueError(f"q_values must be a 2D array; got shape {q_values.shape}")
    N, Q = q_values.shape
    if Q < 2:
        raise ValueError("q_values must contain at least two quantiles")
    if q_levels.shape != (Q,):
        raise ValueError(f"q_levels must have shape ({Q},); got {q_levels.shape}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if thresholds.ndim == 0:
        thresholds = np.full(N, thresholds)
    else:
        thresholds = thresholds.reshape(-1)
    if thresholds.shape[0] != N:
        raise ValueError(
            f"thresholds must be scalar or have length {N}; got {thresholds.shape[0]}"
        )

    torch, selected_device = cuda_device(device)
    if torch is not None:
        return _interpolate_cuda(
            torch,
            selected_device,
            q_values,
            q_levels,
            thresholds,
            chunk_size,
        )

    # NumPy has no row-wise searchsorted. Counting is fully vectorized and Q is
    # small for quantile models, making this much faster than a Python row loop.
    idx = np.count_nonzero(q_values <= thresholds[:, np.newaxis], axis=1)
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


def _interpolate_cuda(torch, device, q_values, q_levels, thresholds, chunk_size):
    # CSV input is normally float64. Preserving it avoids moving interpolation
    # boundaries and changing searchsorted results near a predicted quantile.
    dtype = torch.float64
    levels = torch.as_tensor(q_levels, dtype=dtype, device=device)
    result = np.empty(q_values.shape[0], dtype=np.result_type(q_values, float))

    with torch.inference_mode():
        for start in range(0, q_values.shape[0], chunk_size):
            end = min(start + chunk_size, q_values.shape[0])
            values = torch.as_tensor(q_values[start:end], dtype=dtype, device=device)
            threshold = torch.as_tensor(
                thresholds[start:end], dtype=dtype, device=device
            ).contiguous()
            idx = torch.searchsorted(
                values, threshold.unsqueeze(1), right=True
            ).squeeze(1)
            idx_clamped = idx.clamp(1, values.shape[1] - 1)
            rows = torch.arange(end - start, device=device)
            x0 = values[rows, idx_clamped - 1]
            x1 = values[rows, idx_clamped]
            y0 = levels[idx_clamped - 1]
            y1 = levels[idx_clamped]
            dx = x1 - x0
            slope = torch.where(dx != 0, (y1 - y0) / dx, torch.zeros_like(dx))
            probs = y0 + (threshold - x0) * slope
            probs = torch.where(idx == 0, levels[0], probs)
            probs = torch.where(idx == values.shape[1], levels[-1], probs)
            result[start:end] = probs.clamp(levels[0], levels[-1]).cpu().numpy()
    return result


def quantile_pit(q_values, q_levels, thresholds, device="auto", chunk_size=262144):
    """Compute PIT values P(Y <= y_i | x_i) with per-sample thresholds.

    Parameters
    ----------
    q_values : (N, Q) array
        Predicted quantile values, sorted along axis=1 (no crossing).
    q_levels : (Q,) array
        Quantile levels (taus), sorted in ascending order.
    thresholds : (N,) array
        Per-sample threshold (actual observed values).
    device : str
        ``"auto"`` selects CUDA when available; ``"cpu"`` forces NumPy.
    chunk_size : int
        Maximum rows transferred to CUDA at once.

    Returns
    -------
    (N,) array
        Estimated CDF value for each sample.
    """
    return _interpolate_linear(q_values, q_levels, thresholds, device, chunk_size)
