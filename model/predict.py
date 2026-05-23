from pathlib import Path

import torch

from .model import GPR_H, GPR_b, GPR_phi, load_model

__all__ = [
    "gpr_H",
    "gpr_b",
    "gpr_phi",
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


def gpr_b(X, device=None):
    """Predict b using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    b : array-like of shape (n_samples,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.b.pt"
    model, _, scaler = load_model(GPR_b, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        b = model(X).mean.cpu().numpy()
    return b


def gpr_phi(X, device=None):
    """Predict phi using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    phi : array-like of shape (n_samples,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.phi.pt"
    model, _, scaler = load_model(GPR_phi, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        phi = model(X).mean.cpu().numpy()
    return phi
