"""Utilities for loading prediction inputs with model batch dimensions."""

import numpy as np
import pandas as pd


def load_batched_features(path, index_col=None, batch_col=None):
    """Return feature values and their original row indices in batch order.

    ``batch_col`` refers to columns in the source CSV, before ``index_col`` is
    applied.  Every combination of batch values must contain the same number
    of feature rows, matching the tensor layout expected by batched models.
    """
    raw_df = pd.read_csv(path)
    X_df = pd.read_csv(path, index_col=index_col if index_col else None)

    try:
        batch_keys = raw_df.iloc[:, batch_col or []]
    except IndexError as exc:
        raise ValueError(
            "--batch-col contains a column outside the input range "
            f"[0, {raw_df.shape[1] - 1}]"
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
            f"same number of rows; got batch sizes {batch_sizes.tolist()}"
        )

    # Stable sorting retains the CSV order within each batch.
    order = np.argsort(batch_ids, kind="stable")
    batch_size = int(batch_sizes[0])
    values = X_df.iloc[order].values.reshape(*batch_shape, batch_size, X_df.shape[1])
    row_indices = order.reshape(*batch_shape, batch_size)
    return values, row_indices
