from pathlib import Path

import torch

from .model import (
    GPR_H,
    CgIndependentMtgpqr_phi,
    CgLmcMtgpqr_H,
    GPR_b,
    GPR_phi,
    load_model,
)

__all__ = [
    "gpr_H",
    "gpr_b",
    "gpr_phi",
    "gpqr_H",
    "gpqr_phi",
]


def gpr_H(X, device=None):
    """Predict H using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    H : MultivariateNormal
        Posterior distribution of H.
        Batch shape is () and event shape is (n_samples,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.H.pt"
    model, _, scaler = load_model(GPR_H, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        H = model(X)
    return H


def gpr_b(X, device=None):
    """Predict b using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    b : MultivariateNormal
        Posterior distribution of b.
        Batch shape is () and event shape is (n_samples,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.b.pt"
    model, _, scaler = load_model(GPR_b, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        b = model(X)
    return b


def gpr_phi(X, device=None):
    """Predict phi using the GPR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    phi : MultivariateNormal
        Posterior distribution of phi.
        Batch shape is () and event shape is (n_samples,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.phi.pt"
    model, _, scaler = load_model(GPR_phi, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        phi = model(X)
    return phi


def gpqr_H(X, device=None):
    """Predict H using the GPQR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    quantiles : tensor of shape (Q,)
        Quantile levels.
    H : MultitaskMultivariateNormal
        Posterior distribution of H.
        Batch shape is () and event shape is (n_samples, Q).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPQR.H.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, _, scaler = load_model(CgLmcMtgpqr_H, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        H = model(X)
    return quantiles, H


def gpqr_phi(X, device=None):
    """Predict phi using the GPQR model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    quantiles : tensor of shape (Q,)
        Quantile levels.
    phi : MultitaskMultivariateNormal
        Posterior distribution of phi.
        Batch shape is () and event shape is (n_samples, Q).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPQR.phi.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, _, scaler = load_model(CgIndependentMtgpqr_phi, model_path, device=device)
    model.to(device)
    model.eval()

    X = torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        phi = model(X)
    return quantiles, phi
