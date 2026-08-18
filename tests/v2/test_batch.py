import csv

import numpy as np

from scripts.v2.train.batch import load_batched_arrays


def test_load_batched_arrays_preserves_multiple_target_columns(tmp_path):
    X_path = tmp_path / "X.csv"
    y_path = tmp_path / "y.csv"

    with X_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch", "feature"])
        writer.writerows([["b", 20], ["a", 10], ["b", 21], ["a", 11]])
    with y_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["first", "second"])
        writer.writerows([[200, 2000], [100, 1000], [210, 2100], [110, 1100]])

    X, y = load_batched_arrays(X_path, y_path, ["first", "second"], batch_col=[0])

    assert X.shape == (2, 2, 2)
    assert y.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        y, [[[200, 2000], [210, 2100]], [[100, 1000], [110, 1100]]]
    )
