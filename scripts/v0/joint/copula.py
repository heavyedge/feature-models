import numpy as np
from backend import cuda_device

__all__ = [
    "empirical_copula",
]


def empirical_copula(
    u_train,
    u_pred,
    chunk_size=1024,
    train_chunk_size=32768,
    device="auto",
    progress=None,
):
    """Estimate joint CDF using empirical copula.

    Parameters
    ----------
    u_train : (N, D) array
        PIT values at training points for D variables.
    u_pred : (M, D) array
        Marginal CDF values at prediction points.
    chunk_size : int
        Prediction rows per comparison chunk.
    train_chunk_size : int
        Training rows per comparison chunk.
    device : str
        ``"auto"`` selects CUDA when available; ``"cpu"`` forces NumPy.

    Returns
    -------
    (M,) array
        Estimated joint probability for each prediction point.
    """
    u_train = np.asarray(u_train)
    u_pred = np.asarray(u_pred)
    if u_train.ndim != 2 or u_pred.ndim != 2:
        raise ValueError("u_train and u_pred must both be 2D arrays")
    if u_train.shape[1] != u_pred.shape[1]:
        raise ValueError("u_train and u_pred must have the same number of variables")
    if u_train.shape[0] == 0:
        raise ValueError("u_train must contain at least one row")
    if chunk_size <= 0 or train_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    torch, selected_device = cuda_device(device)
    if torch is not None:
        return _empirical_copula_cuda(
            torch,
            selected_device,
            u_train,
            u_pred,
            chunk_size,
            train_chunk_size,
            progress,
        )

    M = u_pred.shape[0]
    result = np.empty(M, dtype=float)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        counts = np.zeros(end - start, dtype=np.int64)
        pred_chunk = u_pred[start:end, np.newaxis, :]
        for train_start in range(0, u_train.shape[0], train_chunk_size):
            train_end = min(train_start + train_chunk_size, u_train.shape[0])
            indicator = u_train[np.newaxis, train_start:train_end, :] <= pred_chunk
            counts += indicator.all(axis=2).sum(axis=1)
        result[start:end] = counts / u_train.shape[0]
        if progress is not None:
            progress(end)
    return result


def _empirical_copula_cuda(
    torch,
    device,
    u_train,
    u_pred,
    chunk_size,
    train_chunk_size,
    progress=None,
):
    # Preserve the float64 values read by pandas: casting PIT values to float32
    # can change an empirical-CDF comparison at equality boundaries.
    dtype = torch.float64
    train = torch.as_tensor(u_train, dtype=dtype, device=device)
    pred = torch.as_tensor(u_pred, dtype=dtype, device=device)
    result = np.empty(u_pred.shape[0], dtype=float)

    with torch.inference_mode():
        for start in range(0, pred.shape[0], chunk_size):
            end = min(start + chunk_size, pred.shape[0])
            counts = torch.zeros(end - start, dtype=torch.int64, device=device)
            pred_chunk = pred[start:end].unsqueeze(1)
            for train_start in range(0, train.shape[0], train_chunk_size):
                train_end = min(train_start + train_chunk_size, train.shape[0])
                counts += (
                    (train[train_start:train_end].unsqueeze(0) <= pred_chunk)
                    .all(dim=2)
                    .sum(dim=1)
                )
            result[start:end] = (
                (counts.to(torch.float64) / train.shape[0]).cpu().numpy()
            )
            if progress is not None:
                progress(end)
    return result
