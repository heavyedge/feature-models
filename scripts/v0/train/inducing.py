import logging
import math

import torch

logger = logging.getLogger(__name__)


def unique_inducing_points_per_fold(X):
    """Return duplicate-free inducing points with a common count per fold.

    The variational distribution has one inducing-point dimension shared by all
    batch/fold dimensions. Rows are therefore deduplicated independently in
    each fold, after which the largest count that every fold can support is
    retained. Observations in ``X`` are left unchanged.
    """
    num_observations, dim = X.shape[-2:]
    num_folds = math.prod(X.shape[:-2])
    flat_folds = X.reshape(num_folds, num_observations, dim)
    unique_folds = [torch.unique(fold, dim=0) for fold in flat_folds]
    num_inducing = min(fold.shape[0] for fold in unique_folds)
    if num_inducing == 0:
        raise ValueError("At least one inducing point is required per fold.")

    # When folds have different duplicate counts, choose evenly-spaced unique
    # points from the larger folds to retain coverage of their feature range.
    selected_folds = []
    for fold in unique_folds:
        if fold.shape[0] == num_inducing:
            selected_folds.append(fold)
        else:
            indices = (
                torch.linspace(0, fold.shape[0] - 1, num_inducing, device=X.device)
                .round()
                .long()
            )
            selected_folds.append(fold.index_select(0, indices))

    if num_inducing < num_observations:
        logger.info(
            "Using %d duplicate-free inducing points per fold (from %d training rows).",
            num_inducing,
            num_observations,
        )
    return torch.stack(selected_folds).reshape(*X.shape[:-2], num_inducing, dim)
