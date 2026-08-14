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
        raise pytest.UsageError("--models-path is required for v0 tests")
    return Path(models_path).resolve()
