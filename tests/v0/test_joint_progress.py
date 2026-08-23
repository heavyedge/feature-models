import subprocess
import sys
from pathlib import Path

import pandas as pd

JOINT_DIR = Path(__file__).parents[2] / "scripts" / "v0" / "joint"


def test_pit_and_marginal_scripts_log_progress(tmp_path):
    y_path = tmp_path / "Y.csv"
    pred_path = tmp_path / "pred.csv"
    pit_output = tmp_path / "pit-output.csv"
    marginal_output = tmp_path / "marginal-output.csv"

    pd.DataFrame({"H": [0.5]}).to_csv(y_path, index=False)
    pd.DataFrame(
        [(0, 0, "H", 0.1, 0, 0.0), (0, 0, "H", 0.9, 0, 1.0)],
        columns=["index", "batch", "target", "quantile", "sample", "value"],
    ).to_csv(pred_path, index=False)

    pit_run = subprocess.run(
        [
            sys.executable,
            str(JOINT_DIR / "write-pit.py"),
            str(y_path),
            str(pred_path),
            "--quantiles",
            "0.1",
            "0.9",
            "--device",
            "cpu",
            "-o",
            str(pit_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    marginal_run = subprocess.run(
        [
            sys.executable,
            str(JOINT_DIR / "write-marginal.py"),
            str(pred_path),
            "--quantiles",
            "0.1",
            "0.9",
            "--threshold",
            "0.5",
            "--device",
            "cpu",
            "-o",
            str(marginal_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pit-output | PIT H: 1/1 (100.0%)" in pit_run.stderr
    assert "marginal-output | Marginal H: 1/1 (100.0%)" in marginal_run.stderr
    assert pd.read_csv(pit_output)["pit"].tolist() == [0.5]
    assert pd.read_csv(marginal_output)["marginal_prob"].tolist() == [0.5]


def test_write_joint_logs_progress_and_ignores_unused_pit_target(tmp_path):
    x_path = tmp_path / "X.csv"
    pit_path = tmp_path / "pit.csv"
    marginal_path = tmp_path / "marginal.csv"
    output_path = tmp_path / "joint.csv"

    pd.DataFrame({"cosine_of_contact_angle": [0.5]}).to_csv(x_path, index=False)
    pd.DataFrame(
        [
            (0, 0, "H", 0, 0.2),
            (0, 0, "phi_1", 0, 0.3),
            (0, 0, "phi_3", 0, 0.1),
            (1, 0, "H", 0, 0.8),
            (1, 0, "phi_1", 0, 0.9),
            (1, 0, "phi_3", 0, 0.1),
        ],
        columns=["index", "batch", "target", "sample", "pit"],
    ).to_csv(pit_path, index=False)
    pd.DataFrame(
        [(0, 0, "H", 0, 0.5), (0, 0, "phi_1", 0, 0.5)],
        columns=["index", "batch", "target", "sample", "marginal_prob"],
    ).to_csv(marginal_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(JOINT_DIR / "write-joint.py"),
            str(x_path),
            str(pit_path),
            str(marginal_path),
            "--device",
            "cpu",
            "-o",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = pd.read_csv(output_path)
    assert output["joint_prob"].tolist() == [0.5]
    assert "joint | Joint probability: 1/1 (100.0%)" in completed.stderr
    assert "Writing joint probabilities" in completed.stderr
    assert "Done" in completed.stderr
