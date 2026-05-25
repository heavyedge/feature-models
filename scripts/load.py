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


def gpr_H(device=None):
    """Return GPR model for H.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model : gpytorch.models.ExactGP
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    scaler : sklearn scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.H.pt"
    model, likelihood, scaler = load_model(GPR_H, model_path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def gpr_b(device=None):
    """Return GPR model for b.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model : gpytorch.models.ExactGP
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    scaler : sklearn scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.b.pt"
    model, likelihood, scaler = load_model(GPR_b, model_path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def gpr_phi(device=None):
    """Return GPR model for phi.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model : gpytorch.models.ExactGP
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    scaler : sklearn scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPR.phi.pt"
    model, likelihood, scaler = load_model(GPR_phi, model_path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def gpqr_H(device=None):
    """Return GPQR model for H.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    quantiles : tensor of shape (Q,)
        Quantile levels.
    model : gpytorch_qr.models.QuantileGP
    likelihood : gpytorch_qr.likelihoods.ALDLikelihood
    scaler : sklearn scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPQR.H.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = load_model(CgLmcMtgpqr_H, model_path, device=device)
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler


def gpqr_phi(device=None):
    """Return GPQR model for phi.

    Parameters
    ----------
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    quantiles : tensor of shape (Q,)
        Quantile levels.
    model : gpytorch_qr.models.QuantileGP
    likelihood : gpytorch_qr.likelihoods.ALDLikelihood
    scaler : sklearn scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(__file__).parent / "GPQR.phi.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = load_model(
        CgIndependentMtgpqr_phi, model_path, device=device
    )
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler
