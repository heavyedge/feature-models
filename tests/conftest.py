import csv
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--models-path",
        action="store",
        default=None,
        help="Add the directory containing the models package to sys.path.",
    )


@pytest.fixture
def models_path(request):
    models_path = request.config.getoption("--models-path")
    if models_path is None:
        raise pytest.UsageError("--models-path is required")
    return Path(models_path).resolve()


@pytest.fixture(scope="session")
def Xtest_path(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "Xtest.csv"
    header = ["feature1", "feature2", "feature3"]
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([1.0, 2.0, 3.0])
    return file_path
