import numpy as np
import pandas as pd

from scripts.v1.train.batch import load_batched_arrays


def test_outputs_are_the_last_gp_batch(tmp_path):
    X_path = tmp_path / "X.csv"
    y_path = tmp_path / "y.csv"
    pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "x": [10, 11, 20, 21],
        }
    ).to_csv(X_path, index=False)
    pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "H": [1, 2, 3, 4],
            "phi_1": [5, 6, 7, 8],
            "phi_3": [9, 10, 11, 12],
        }
    ).to_csv(y_path, index=False)

    X, y = load_batched_arrays(
        X_path,
        y_path,
        ("H", "phi_1", "phi_3"),
        index_col=[0],
        batch_col=[0],
    )

    assert X.shape == (2, 2, 1)
    assert y.shape == (2, 3, 2)
    np.testing.assert_array_equal(y[0], [[1, 2], [5, 6], [9, 10]])
