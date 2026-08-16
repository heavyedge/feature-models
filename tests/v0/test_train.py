from pathlib import Path

import pytest
import torch


@pytest.fixture
def select_inducing_points(monkeypatch):
    train_scripts = Path(__file__).parents[2] / "scripts" / "v0" / "train"
    monkeypatch.syspath_prepend(str(train_scripts))
    from inducing import unique_inducing_points_per_fold

    return unique_inducing_points_per_fold


def assert_unique_rows_per_fold(inducing_points):
    num_inducing, dim = inducing_points.shape[-2:]
    folds = inducing_points.reshape(-1, num_inducing, dim)
    assert all(torch.unique(fold, dim=0).shape[0] == num_inducing for fold in folds)


def test_unique_inducing_points_without_batch(select_inducing_points):
    X = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
    original_X = X.clone()

    inducing_points = select_inducing_points(X)

    assert inducing_points.shape == (3, 2)
    assert_unique_rows_per_fold(inducing_points)
    torch.testing.assert_close(X, original_X)


def test_unique_inducing_points_use_common_count_across_folds(
    select_inducing_points,
):
    X = torch.tensor(
        [
            [[0.0], [0.0], [1.0], [2.0], [3.0]],
            [[4.0], [5.0], [4.0], [5.0], [4.0]],
        ]
    )

    inducing_points = select_inducing_points(X)

    # The folds contain four and two unique rows respectively, so the shared
    # variational inducing dimension must be two.
    assert inducing_points.shape == (2, 2, 1)
    assert_unique_rows_per_fold(inducing_points)
    for fold, inducing_fold in zip(X, inducing_points):
        assert all(
            torch.any(torch.all(fold == inducing_point, dim=-1))
            for inducing_point in inducing_fold
        )


def test_unique_inducing_points_preserve_multiple_batch_dimensions(
    select_inducing_points,
):
    X = torch.tensor(
        [
            [
                [[0.0], [0.0], [1.0], [2.0]],
                [[3.0], [4.0], [3.0], [4.0]],
            ],
            [
                [[5.0], [6.0], [7.0], [8.0]],
                [[9.0], [9.0], [10.0], [11.0]],
            ],
        ]
    )

    inducing_points = select_inducing_points(X)

    assert inducing_points.shape == (2, 2, 2, 1)
    assert_unique_rows_per_fold(inducing_points)


def test_unique_inducing_points_reject_empty_folds(select_inducing_points):
    X = torch.empty(2, 0, 3)

    with pytest.raises(ValueError, match="At least one inducing point"):
        select_inducing_points(X)
