import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/v1/joint/write-pit.gpr.py"


def test_gpr_pit_uses_predictive_gaussian(tmp_path):
    target_path = tmp_path / "target.csv"
    prediction_path = tmp_path / "prediction.csv"
    out_path = tmp_path / "pit.csv"

    pd.DataFrame({"y": [0.0, 2.5]}).to_csv(target_path, index=False)
    pd.DataFrame(
        {
            "index": [0, 1],
            "batch": [np.nan, np.nan],
            "target": ["y", "y"],
            "latent_mean": [0.0, 2.0],
            "latent_std": [0.5, 0.25],
            "predictive_mean": [0.0, 2.0],
            "predictive_std": [1.0, 0.5],
        }
    ).to_csv(prediction_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(target_path),
            str(prediction_path),
            "--quantiles",
            "0.05",
            "0.5",
            "0.95",
            "--out",
            str(out_path),
        ],
        check=True,
    )

    result = pd.read_csv(out_path)
    assert result.columns.tolist() == ["index", "batch", "target", "sample", "pit"]
    assert result["target"].tolist() == ["y", "y"]
    assert result["sample"].tolist() == [0, 0]
    np.testing.assert_allclose(result["pit"], [0.5, 0.8413447460685429])
