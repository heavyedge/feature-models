import numpy as np
from backend import cuda_device

__all__ = [
    "quantile_density",
    "quantile_interpolation",
    "quantile_log_density",
    "quantile_pit",
]


def quantile_interpolation(
    q_values,
    q_levels,
    threshold,
    device="auto",
    chunk_size=262144,
    progress=None,
):
    """Estimate P(Y <= threshold) from predicted quantiles.

    A monotone PCHIP interpolates the CDF between predicted quantiles. Beyond
    the outer quantiles, exponential tails are joined with matching value and
    first derivative, making the resulting CDF continuously differentiable.

    Parameters
    ----------
    q_values : (N, Q) array
        Predicted quantile values, strictly increasing along axis=1.
    q_levels : (Q,) array
        Quantile levels, strictly increasing and in (0, 1).
    threshold : float or (N,) array
        Threshold value(s).
    device : str
        ``"auto"`` selects CUDA when available; ``"cpu"`` forces NumPy.
    chunk_size : int
        Maximum rows transferred to CUDA at once.
    """
    return _interpolate_pchip(
        q_values, q_levels, threshold, device, chunk_size, progress
    )


def quantile_density(
    q_values,
    q_levels,
    observations,
    device="auto",
    chunk_size=262144,
    progress=None,
):
    """Evaluate the density implied by predicted quantiles.

    The density is the derivative of :func:`quantile_interpolation`: a
    monotone PCHIP density between the predicted quantiles and exponential
    densities beyond the two outer quantiles.
    """
    q_values, q_levels, observations = _validate_inputs(
        q_values, q_levels, observations, chunk_size
    )
    torch, selected_device = cuda_device(device)
    if torch is not None:
        return _density_cuda(
            torch,
            selected_device,
            q_values,
            q_levels,
            observations,
            chunk_size,
            progress,
        )
    return _density_numpy(q_values, q_levels, observations, progress)


def quantile_log_density(q_values, q_levels, observations, progress=None):
    """Evaluate the log-density implied by predicted quantiles on the CPU.

    This is evaluated directly in log space in the exponential tails, avoiding
    underflow for observations far outside the predicted quantile range.
    """
    q_values, q_levels, observations = _validate_inputs(
        q_values, q_levels, observations, chunk_size=1
    )
    return _log_density_numpy(q_values, q_levels, observations, progress)


def _validate_inputs(q_values, q_levels, thresholds, chunk_size):
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
    if not np.all(np.isfinite(q_levels)) or not np.all((q_levels > 0) & (q_levels < 1)):
        raise ValueError("q_levels must be finite and in (0, 1)")
    if np.any(np.diff(q_levels) <= 0):
        raise ValueError("q_levels must be strictly increasing")
    if not np.all(np.isfinite(q_values)):
        raise ValueError("q_values must be finite")
    if np.any(np.diff(q_values, axis=1) <= 0):
        raise ValueError("q_values must be strictly increasing along axis 1")
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
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must be finite")
    return q_values, q_levels, thresholds


def _interpolate_pchip(q_values, q_levels, thresholds, device, chunk_size, progress):
    q_values, q_levels, thresholds = _validate_inputs(
        q_values, q_levels, thresholds, chunk_size
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
            progress,
        )
    return _interpolate_numpy(q_values, q_levels, thresholds, progress)


def _endpoint_derivatives_numpy(values, level_diffs):
    """PCHIP endpoint derivatives, with positive secant fallbacks."""
    first_delta = level_diffs[0] / (values[:, 1] - values[:, 0])
    last_delta = level_diffs[-1] / (values[:, -1] - values[:, -2])
    if values.shape[1] == 2:
        return first_delta, last_delta

    h0 = values[:, 1] - values[:, 0]
    h1 = values[:, 2] - values[:, 1]
    delta1 = level_diffs[1] / h1
    first = ((2 * h0 + h1) * first_delta - h0 * delta1) / (h0 + h1)
    first = np.where(first > 0, first, first_delta)

    hn = values[:, -1] - values[:, -2]
    hp = values[:, -2] - values[:, -3]
    delta_prev = level_diffs[-2] / hp
    last = ((2 * hn + hp) * last_delta - hn * delta_prev) / (hn + hp)
    last = np.where(last > 0, last, last_delta)
    return first, last


def _knot_derivative_numpy(values, level_diffs, knot, first, last):
    """Evaluate PCHIP derivatives only at the requested knot in each row."""
    if values.shape[1] == 2:
        return first

    rows = np.arange(values.shape[0])
    interior = np.clip(knot, 1, values.shape[1] - 2)
    h_prev = values[rows, interior] - values[rows, interior - 1]
    h_next = values[rows, interior + 1] - values[rows, interior]
    delta_prev = level_diffs[interior - 1] / h_prev
    delta_next = level_diffs[interior] / h_next
    w1 = 2 * h_next + h_prev
    w2 = h_next + 2 * h_prev
    derivative = (w1 + w2) / (w1 / delta_prev + w2 / delta_next)
    derivative = np.where(knot == 0, first, derivative)
    return np.where(knot == values.shape[1] - 1, last, derivative)


def _interpolate_numpy(values, levels, thresholds, progress):
    N, Q = values.shape
    level_diffs = np.diff(levels)

    # Counting avoids constructing one scipy interpolator per sample.
    idx = np.count_nonzero(values <= thresholds[:, np.newaxis], axis=1)
    interval = np.clip(idx - 1, 0, Q - 2)
    rows = np.arange(N)
    x0 = values[rows, interval]
    x1 = values[rows, interval + 1]
    y0 = levels[interval]
    y1 = levels[interval + 1]
    h = x1 - x0

    first, last = _endpoint_derivatives_numpy(values, level_diffs)
    d0 = _knot_derivative_numpy(values, level_diffs, interval, first, last)
    d1 = _knot_derivative_numpy(values, level_diffs, interval + 1, first, last)

    # Clipping avoids polynomial overflow for observations far into a tail.
    t = np.clip((thresholds - x0) / h, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    if Q == 2:
        # PCHIP is exactly linear here; this form also preserves the previous
        # implementation's rounding at simple midpoints.
        probs = y0 + t * (y1 - y0)
    else:
        probs = (
            (2 * t3 - 3 * t2 + 1) * y0
            + (t3 - 2 * t2 + t) * h * d0
            + (-2 * t3 + 3 * t2) * y1
            + (t3 - t2) * h * d1
        )
    probs = np.clip(probs, y0, y1)

    left = levels[0] * np.exp(
        np.minimum((thresholds - values[:, 0]) * first / levels[0], 0.0)
    )
    upper_distance = (thresholds - values[:, -1]) * last / (1.0 - levels[-1])
    right = levels[-1] - (1.0 - levels[-1]) * np.expm1(-np.maximum(upper_distance, 0.0))
    probs = np.where(idx == 0, left, probs)
    probs = np.where(idx == Q, right, probs)

    if progress is not None:
        progress(N)
    return np.clip(probs, 0.0, 1.0)


def _density_numpy(values, levels, observations, progress):
    N, Q = values.shape
    level_diffs = np.diff(levels)
    idx = np.count_nonzero(values <= observations[:, np.newaxis], axis=1)
    interval = np.clip(idx - 1, 0, Q - 2)
    rows = np.arange(N)
    x0 = values[rows, interval]
    x1 = values[rows, interval + 1]
    y0 = levels[interval]
    y1 = levels[interval + 1]
    h = x1 - x0

    first, last = _endpoint_derivatives_numpy(values, level_diffs)
    d0 = _knot_derivative_numpy(values, level_diffs, interval, first, last)
    d1 = _knot_derivative_numpy(values, level_diffs, interval + 1, first, last)

    t = np.clip((observations - x0) / h, 0.0, 1.0)
    if Q == 2:
        density = (y1 - y0) / h
    else:
        t2 = t * t
        density = (
            (6 * t2 - 6 * t) * y0
            + (3 * t2 - 4 * t + 1) * h * d0
            + (-6 * t2 + 6 * t) * y1
            + (3 * t2 - 2 * t) * h * d1
        ) / h

    left = first * np.exp(
        np.minimum((observations - values[:, 0]) * first / levels[0], 0.0)
    )
    upper_distance = (observations - values[:, -1]) * last / (1.0 - levels[-1])
    right = last * np.exp(-np.maximum(upper_distance, 0.0))
    density = np.where(idx == 0, left, density)
    density = np.where(idx == Q, right, density)

    if progress is not None:
        progress(N)
    return np.maximum(density, 0.0)


def _log_density_numpy(values, levels, observations, progress):
    N, Q = values.shape
    level_diffs = np.diff(levels)
    idx = np.count_nonzero(values <= observations[:, np.newaxis], axis=1)
    interval = np.clip(idx - 1, 0, Q - 2)
    rows = np.arange(N)
    x0 = values[rows, interval]
    x1 = values[rows, interval + 1]
    y0 = levels[interval]
    y1 = levels[interval + 1]
    h = x1 - x0

    first, last = _endpoint_derivatives_numpy(values, level_diffs)
    d0 = _knot_derivative_numpy(values, level_diffs, interval, first, last)
    d1 = _knot_derivative_numpy(values, level_diffs, interval + 1, first, last)

    t = np.clip((observations - x0) / h, 0.0, 1.0)
    if Q == 2:
        density = (y1 - y0) / h
    else:
        t2 = t * t
        density = (
            (6 * t2 - 6 * t) * y0
            + (3 * t2 - 4 * t + 1) * h * d0
            + (-6 * t2 + 6 * t) * y1
            + (3 * t2 - 2 * t) * h * d1
        ) / h

    with np.errstate(divide="ignore"):
        log_density = np.log(np.maximum(density, 0.0))
    left = np.log(first) + (observations - values[:, 0]) * first / levels[0]
    upper_distance = (observations - values[:, -1]) * last / (1.0 - levels[-1])
    right = np.log(last) - upper_distance
    log_density = np.where(idx == 0, left, log_density)
    log_density = np.where(idx == Q, right, log_density)

    if progress is not None:
        progress(N)
    return log_density


def _endpoint_derivatives_torch(torch, values, level_diffs):
    first_delta = level_diffs[0] / (values[:, 1] - values[:, 0])
    last_delta = level_diffs[-1] / (values[:, -1] - values[:, -2])
    if values.shape[1] == 2:
        return first_delta, last_delta

    h0 = values[:, 1] - values[:, 0]
    h1 = values[:, 2] - values[:, 1]
    delta1 = level_diffs[1] / h1
    first = ((2 * h0 + h1) * first_delta - h0 * delta1) / (h0 + h1)
    first = torch.where(first > 0, first, first_delta)

    hn = values[:, -1] - values[:, -2]
    hp = values[:, -2] - values[:, -3]
    delta_prev = level_diffs[-2] / hp
    last = ((2 * hn + hp) * last_delta - hn * delta_prev) / (hn + hp)
    last = torch.where(last > 0, last, last_delta)
    return first, last


def _knot_derivative_torch(values, level_diffs, knot, first, last):
    if values.shape[1] == 2:
        return first

    interior = knot.clamp(1, values.shape[1] - 2)
    x = values.gather(1, interior.unsqueeze(1)).squeeze(1)
    x_prev = values.gather(1, (interior - 1).unsqueeze(1)).squeeze(1)
    x_next = values.gather(1, (interior + 1).unsqueeze(1)).squeeze(1)
    h_prev = x - x_prev
    h_next = x_next - x
    delta_prev = level_diffs[interior - 1] / h_prev
    delta_next = level_diffs[interior] / h_next
    w1 = 2 * h_next + h_prev
    w2 = h_next + 2 * h_prev
    derivative = (w1 + w2) / (w1 / delta_prev + w2 / delta_next)
    derivative = derivative.where(knot != 0, first)
    return derivative.where(knot != values.shape[1] - 1, last)


def _interpolate_cuda(
    torch, device, q_values, q_levels, thresholds, chunk_size, progress=None
):
    dtype = torch.float64
    levels = torch.as_tensor(q_levels, dtype=dtype, device=device)
    level_diffs = levels[1:] - levels[:-1]
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
            interval = (idx - 1).clamp(0, values.shape[1] - 2)

            x0 = values.gather(1, interval.unsqueeze(1)).squeeze(1)
            x1 = values.gather(1, (interval + 1).unsqueeze(1)).squeeze(1)
            y0 = levels[interval]
            y1 = levels[interval + 1]
            h = x1 - x0

            first, last = _endpoint_derivatives_torch(torch, values, level_diffs)
            d0 = _knot_derivative_torch(values, level_diffs, interval, first, last)
            d1 = _knot_derivative_torch(values, level_diffs, interval + 1, first, last)

            t = ((threshold - x0) / h).clamp(0.0, 1.0)
            t2 = t * t
            t3 = t2 * t
            if values.shape[1] == 2:
                probs = y0 + t * (y1 - y0)
            else:
                probs = (
                    (2 * t3 - 3 * t2 + 1) * y0
                    + (t3 - 2 * t2 + t) * h * d0
                    + (-2 * t3 + 3 * t2) * y1
                    + (t3 - t2) * h * d1
                )
            probs = probs.clamp(y0, y1)

            left = levels[0] * torch.exp(
                ((threshold - values[:, 0]) * first / levels[0]).clamp(max=0.0)
            )
            upper_distance = (threshold - values[:, -1]) * last / (1.0 - levels[-1])
            right = levels[-1] - (1.0 - levels[-1]) * torch.expm1(
                -upper_distance.clamp(min=0.0)
            )
            probs = torch.where(idx == 0, left, probs)
            probs = torch.where(idx == values.shape[1], right, probs)
            result[start:end] = probs.clamp(0.0, 1.0).cpu().numpy()
            if progress is not None:
                progress(end)
    return result


def _density_cuda(
    torch, device, q_values, q_levels, observations, chunk_size, progress=None
):
    dtype = torch.float64
    levels = torch.as_tensor(q_levels, dtype=dtype, device=device)
    level_diffs = levels[1:] - levels[:-1]
    result = np.empty(q_values.shape[0], dtype=np.result_type(q_values, float))

    with torch.inference_mode():
        for start in range(0, q_values.shape[0], chunk_size):
            end = min(start + chunk_size, q_values.shape[0])
            values = torch.as_tensor(q_values[start:end], dtype=dtype, device=device)
            observation = torch.as_tensor(
                observations[start:end], dtype=dtype, device=device
            ).contiguous()
            idx = torch.searchsorted(
                values, observation.unsqueeze(1), right=True
            ).squeeze(1)
            interval = (idx - 1).clamp(0, values.shape[1] - 2)

            x0 = values.gather(1, interval.unsqueeze(1)).squeeze(1)
            x1 = values.gather(1, (interval + 1).unsqueeze(1)).squeeze(1)
            y0 = levels[interval]
            y1 = levels[interval + 1]
            h = x1 - x0
            first, last = _endpoint_derivatives_torch(torch, values, level_diffs)
            d0 = _knot_derivative_torch(values, level_diffs, interval, first, last)
            d1 = _knot_derivative_torch(values, level_diffs, interval + 1, first, last)

            t = ((observation - x0) / h).clamp(0.0, 1.0)
            if values.shape[1] == 2:
                density = (y1 - y0) / h
            else:
                t2 = t * t
                density = (
                    (6 * t2 - 6 * t) * y0
                    + (3 * t2 - 4 * t + 1) * h * d0
                    + (-6 * t2 + 6 * t) * y1
                    + (3 * t2 - 2 * t) * h * d1
                ) / h

            left = first * torch.exp(
                ((observation - values[:, 0]) * first / levels[0]).clamp(max=0.0)
            )
            upper_distance = (observation - values[:, -1]) * last / (1.0 - levels[-1])
            right = last * torch.exp(-upper_distance.clamp(min=0.0))
            density = torch.where(idx == 0, left, density)
            density = torch.where(idx == values.shape[1], right, density)
            result[start:end] = density.clamp(min=0.0).cpu().numpy()
            if progress is not None:
                progress(end)
    return result


def quantile_pit(
    q_values,
    q_levels,
    thresholds,
    device="auto",
    chunk_size=262144,
    progress=None,
):
    """Compute smooth PIT values P(Y <= y_i | x_i).

    See :func:`quantile_interpolation` for the PCHIP and tail construction.
    """
    return _interpolate_pchip(
        q_values, q_levels, thresholds, device, chunk_size, progress
    )
