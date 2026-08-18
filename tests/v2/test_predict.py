import subprocess
import sys


def test_predict_priormean(models_path, Xtest_path, tmp_path):
    output_path = tmp_path / "predict.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_models.predict-prior_mean",
            str(Xtest_path),
            "--out",
            str(output_path),
        ],
        check=True,
        cwd=models_path,
    )
    assert output_path.exists()


def test_predict_gpr(models_path, Xtest_path, tmp_path):
    output_path = tmp_path / "predict-gpr.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_models.predict-gpr",
            str(Xtest_path),
            "--out",
            str(output_path),
        ],
        check=True,
        cwd=models_path,
    )
    assert output_path.exists()
