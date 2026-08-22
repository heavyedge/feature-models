import subprocess
import sys

import pandas as pd

TARGETS = {"H", "phi_1", "phi_3"}


def run_prediction(module, models_path, Xtest_path, output_path, *args):
    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            str(Xtest_path),
            *args,
            "--out",
            str(output_path),
        ],
        check=True,
        cwd=models_path,
    )
    result = pd.read_csv(output_path)
    assert set(result["target"]) == TARGETS
    return result


def test_predict_priormean(models_path, Xtest_path, tmp_path):
    result = run_prediction(
        "feature_models.predict-prior_mean",
        models_path,
        Xtest_path,
        tmp_path / "prior.csv",
    )
    assert len(result) == 3


def test_predict_gpr(models_path, Xtest_path, tmp_path):
    result = run_prediction(
        "feature_models.predict-gpr",
        models_path,
        Xtest_path,
        tmp_path / "gpr.csv",
    )
    assert len(result) == 3


def test_predict_gpqr(models_path, Xtest_path, tmp_path):
    result = run_prediction(
        "feature_models.predict-gpqr",
        models_path,
        Xtest_path,
        tmp_path / "gpqr.csv",
        "--num-samples",
        "2",
    )
    assert len(result) == 2 * 3 * result["quantile"].nunique()


def test_predict_with_additional_input_batch(models_path, tmp_path):
    X_path = tmp_path / "X_batched.csv"
    pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "Rgt": [1.0, 1.2, 1.0, 1.2],
            "Ca": [0.5, 0.5, 0.5, 0.5],
            "cos_theta": [0.8, 0.8, 0.8, 0.8],
        }
    ).to_csv(X_path, index=False)

    result = run_prediction(
        "feature_models.predict-gpqr",
        models_path,
        X_path,
        tmp_path / "gpqr_batched.csv",
        "--index-col",
        "0",
        "--batch-col",
        "0",
        "--num-samples",
        "1",
    )

    assert set(result["batch"]) == {0, 1}
    assert len(result) == 2 * 3 * 2 * result["quantile"].nunique()
