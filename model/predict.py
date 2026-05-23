from pathlib import Path

import torch

from .model import GPR_H, load_model

__all__ = [
    "gpr_H",
]


def gpr_H(X, device=None):
    """Predict H using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    H : array-like of shape (n_samples,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.H.pt"
    model, _, scaler = load_model(GPR_H, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        H = model(X).mean.cpu().numpy()
    return H
