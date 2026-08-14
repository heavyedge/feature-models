import subprocess
import sys


def test_predict_priormean_H(models_path, Xtest_path, tmp_path):
    output_path = tmp_path / "predicted_H.csv"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_models.predict-prior_mean",
            str(Xtest_path),
            "--target",
            "H",
            "--out",
            str(output_path),
        ],
        check=True,
        cwd=models_path,
    )
    # assert output_path.exists()
