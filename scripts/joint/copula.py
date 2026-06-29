import numpy as np

__all__ = [
    "empirical_copula",
]


def empirical_copula(u_train, u_pred, chunk_size=4096):
    """Estimate joint CDF using empirical copula.

    Parameters
    ----------
    u_train : (N, D) array
        PIT values at training points for D variables.
    u_pred : (M, D) array
        Marginal CDF values at prediction points.
    chunk_size : int
        Process predictions in chunks to limit memory.

    Returns
    -------
    (M,) array
        Estimated joint probability for each prediction point.
    """
    M = u_pred.shape[0]
    result = np.empty(M)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        indicator = u_train[np.newaxis, :, :] <= u_pred[start:end, np.newaxis, :]
        result[start:end] = indicator.all(axis=2).mean(axis=1)
    return result
