import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/v1/model_selection/likelihood.py"
)


def run_likelihood(tmp_path, prediction, target, prediction_type, *extra_args):
    pred_path = tmp_path / "prediction.csv"
    target_path = tmp_path / "target.csv"
    out_path = tmp_path / "likelihood.csv"
    prediction.to_csv(pred_path, index=False)
    target.to_csv(target_path, index=False)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(pred_path),
            str(target_path),
            "--type",
            prediction_type,
            *extra_args,
            "--out",
            str(out_path),
        ],
        check=True,
    )
    return pd.read_csv(out_path)


def test_gpr_likelihood_uses_predictive_gaussian(tmp_path):
    prediction = pd.DataFrame(
        {
            "index": [0, 1],
            "target": ["y", "y"],
            "predictive_mean": [0.0, 2.0],
            "predictive_std": [1.0, 0.5],
        }
    )
    target = pd.DataFrame({"y": [0.0, 2.5]})

    result = run_likelihood(tmp_path, prediction, target, "GPR")

    expected = np.array([1 / np.sqrt(2 * np.pi), np.exp(-0.5) / np.sqrt(0.5 * np.pi)])
    np.testing.assert_allclose(result["likelihood"], expected)
    np.testing.assert_allclose(result["negative_log_likelihood"], -np.log(expected))


def test_gpqr_likelihood_averages_posterior_sample_densities(tmp_path):
    prediction = pd.DataFrame.from_records(
        {
            "index": index,
            "target": "y",
            "quantile": quantile,
            "sample": sample,
            "value": value,
        }
        for index in range(3)
        for quantile, sample, value in (
            (0.25, 0, -1.0),
            (0.25, 1, 1.0),
            (0.75, 0, 1.0),
            (0.75, 1, 3.0),
        )
    )
    target = pd.DataFrame({"y": [1.0, -1.0, 3.0]})

    result = run_likelihood(
        tmp_path,
        prediction,
        target,
        "GPQR",
        "--quantile-levels",
        "0.25",
        "0.75",
    )

    # The posterior samples have quantiles [-1, 1] and [1, 3].  At the center
    # both densities are 0.25; in either tail their mixture density is the mean
    # of 0.25 and 0.25 * exp(-2).
    tail_mixture = 0.125 * (1 + np.exp(-2))
    expected = np.array([0.25, tail_mixture, tail_mixture])
    np.testing.assert_allclose(result["likelihood"], expected)
    np.testing.assert_allclose(result["negative_log_likelihood"], -np.log(expected))
