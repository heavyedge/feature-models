import sys
from pathlib import Path

import numpy as np
import pytest

JOINT_DIR = Path(__file__).parents[2] / "scripts" / "v0" / "joint"
sys.path.insert(0, str(JOINT_DIR))

from copula import _empirical_copula_cuda, empirical_copula  # noqa: E402
from pit import _interpolate_cuda, quantile_interpolation, quantile_pit  # noqa: E402


def reference_pit(q_values, q_levels, thresholds):
    thresholds = np.broadcast_to(thresholds, (len(q_values),))
    result = []
    for values, threshold in zip(q_values, thresholds):
        idx = np.searchsorted(values, threshold, side="right")
        if idx == 0:
            result.append(q_levels[0])
        elif idx == len(q_levels):
            result.append(q_levels[-1])
        else:
            x0, x1 = values[idx - 1 : idx + 1]
            y0, y1 = q_levels[idx - 1 : idx + 1]
            slope = 0.0 if x0 == x1 else (y1 - y0) / (x1 - x0)
            result.append(
                np.clip(y0 + (threshold - x0) * slope, q_levels[0], q_levels[-1])
            )
    return np.asarray(result)


def test_pit_cpu_matches_rowwise_reference():
    values = np.array(
        [[0.0, 1.0, 2.0], [1.0, 3.0, 5.0], [2.0, 2.0, 4.0], [-3.0, 0.0, 7.0]]
    )
    levels = np.array([0.1, 0.5, 0.9])
    thresholds = np.array([-1.0, 4.0, 2.0, 8.0])

    expected = reference_pit(values, levels, thresholds)
    np.testing.assert_allclose(
        quantile_pit(values, levels, thresholds, device="cpu", chunk_size=2),
        expected,
    )
    np.testing.assert_allclose(
        quantile_interpolation(values, levels, 1.5, device="cpu"),
        reference_pit(values, levels, 1.5),
    )


def test_empirical_copula_cpu_matches_direct_comparison_across_chunks():
    rng = np.random.default_rng(42)
    train = rng.uniform(size=(17, 3))
    pred = rng.uniform(size=(11, 3))
    expected = (
        (train[np.newaxis, :, :] <= pred[:, np.newaxis, :]).all(axis=2).mean(axis=1)
    )

    actual = empirical_copula(
        train,
        pred,
        chunk_size=4,
        train_chunk_size=5,
        device="cpu",
    )
    np.testing.assert_array_equal(actual, expected)


def test_torch_kernels_match_numpy_on_cpu():
    torch = pytest.importorskip("torch")
    values = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]])
    levels = np.array([0.1, 0.5, 0.9])
    thresholds = np.array([1.5, 4.0])
    expected_pit = quantile_pit(values, levels, thresholds, device="cpu")
    actual_pit = _interpolate_cuda(
        torch, torch.device("cpu"), values, levels, thresholds, chunk_size=1
    )
    np.testing.assert_allclose(actual_pit, expected_pit)

    train = np.array([[0.1, 0.2], [0.5, 0.4], [0.8, 0.9]])
    pred = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])
    expected_copula = empirical_copula(train, pred, device="cpu")
    actual_copula = _empirical_copula_cuda(
        torch,
        torch.device("cpu"),
        train,
        pred,
        chunk_size=2,
        train_chunk_size=2,
    )
    np.testing.assert_array_equal(actual_copula, expected_copula)


def test_invalid_chunk_sizes_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        empirical_copula(np.ones((1, 2)), np.ones((1, 2)), chunk_size=0)
    with pytest.raises(ValueError, match="positive"):
        quantile_pit(np.ones((1, 2)), np.array([0.1, 0.9]), [1.0], chunk_size=0)
