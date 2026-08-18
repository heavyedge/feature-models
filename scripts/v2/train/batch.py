"""Load aligned feature and target CSV files into batched NumPy arrays."""

import numpy as np
import pandas as pd


def load_batched_arrays(X_path, y_path, target, index_col=None, batch_col=None):
    """Return X and y arrays with one batch dimension per ``batch_col``.

    Batch columns are selected from the raw X CSV.  Index columns are removed
    from both CSV files before their values are returned.  The returned y array
    retains its target dimension as the final axis.
    """
    X_raw_df = pd.read_csv(X_path)
    X_df = pd.read_csv(X_path, index_col=index_col if index_col else None)
    y_df = pd.read_csv(y_path, index_col=index_col if index_col else None)

    if len(X_df) != len(y_df):
        raise ValueError("X and y must contain the same number of rows.")
    if index_col and not X_df.index.equals(y_df.index):
        raise ValueError("X and y index columns must identify rows in the same order.")

    try:
        batch_keys = X_raw_df.iloc[:, batch_col or []]
    except IndexError as exc:
        raise ValueError(
            "--batch-col contains a column outside the input range "
            f"[0, {X_raw_df.shape[1] - 1}]"
        ) from exc

    batch_codes = []
    batch_shape = []
    for column_index in range(batch_keys.shape[1]):
        codes, levels = pd.factorize(batch_keys.iloc[:, column_index], sort=False)
        if (codes == -1).any():
            codes = np.where(codes == -1, len(levels), codes)
            batch_shape.append(len(levels) + 1)
        else:
            batch_shape.append(len(levels))
        batch_codes.append(codes)

    batch_shape = tuple(batch_shape)
    batch_ids = np.zeros(len(X_df), dtype=int)
    num_batches = 1
    for codes, size in zip(reversed(batch_codes), reversed(batch_shape)):
        batch_ids += codes * num_batches
        num_batches *= size

    batch_sizes = np.bincount(batch_ids, minlength=num_batches)
    if batch_shape and ((batch_sizes == 0).any() or len(set(batch_sizes)) != 1):
        raise ValueError(
            "--batch-col must define a complete grid whose every batch has the "
            "same number of rows; "
            f"got batch sizes {batch_sizes.tolist()}"
        )

    order = np.argsort(batch_ids, kind="stable")
    batch_size = int(batch_sizes[0])
    X = X_df.iloc[order].values.reshape(*batch_shape, batch_size, X_df.shape[1])
    target_names = [target] if isinstance(target, str) else list(target)
    y = (
        y_df[target_names]
        .iloc[order]
        .values.reshape(*batch_shape, batch_size, len(target_names))
    )
    return X, y
