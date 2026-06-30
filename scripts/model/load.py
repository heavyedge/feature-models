from pathlib import Path

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP, QuantileGP

from .gpqr import (
    CgIndependentMtgpqr_phi,
    CgLmcMtgpqr_H,
)
from .gpr import (
    GPR_H,
    GPR_b,
    GPR_phi,
)

__all__ = [
    "load_mean_H",
    "load_mean_b",
    "load_mean_phi",
    "load_gpqr_H",
    "load_gpqr_phi",
]


def _load_model(model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if issubclass(model_class, ExactGP):
        likelihood = GaussianLikelihood()
    elif issubclass(model_class, CenterGapQuantileGP):
        likelihood = MultitaskCenterGapQuantileGPLikelihood(
            checkpoint["quantiles"],
            checkpoint["num_lower_quantiles"],
        )
    elif issubclass(model_class, DirectQuantileGP):
        likelihood = MultitaskQuantileGPLikelihood(
            checkpoint["quantiles"],
        )
    else:
        raise ValueError("Unsupported model class.")
    if issubclass(model_class, ExactGP):
        model = model_class(
            train_x=checkpoint["train_x"],
            train_y=checkpoint["train_y"],
            likelihood=likelihood,
        )
    if issubclass(model_class, QuantileGP):
        model = model_class(
            inducing_points=checkpoint["inducing_points"],
            num_quantiles=len(checkpoint["quantiles"]),
            num_lower_quantiles=checkpoint["num_lower_quantiles"],
            num_latents=checkpoint["num_latents"],
            num_lower_latents=checkpoint["num_lower_latents"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    if device is not None:
        model.to(device)
        likelihood.to(device)
    scaler = checkpoint["scaler"]
    return model, likelihood, scaler


def load_mean_H(path=None, device=None):
    """Return GPR model for H.

    Parameters
    ----------
    path : str or Path, optional
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

    if path is None:
        path = Path(__file__).parent / "mean.H.pt"
    model, likelihood, scaler = _load_model(GPR_H, path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def load_mean_b(path=None, device=None):
    """Return GPR model for b.

    Parameters
    ----------
    path : str or Path, optional
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

    if path is None:
        path = Path(__file__).parent / "mean.b.pt"
    model, likelihood, scaler = _load_model(GPR_b, path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def load_mean_phi(path=None, device=None):
    """Return GPR model for phi.

    Parameters
    ----------
    path : str or Path, optional
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

    if path is None:
        path = Path(__file__).parent / "mean.phi.pt"
    model, likelihood, scaler = _load_model(GPR_phi, path, device=device)
    model.to(device)
    model.eval()
    return model, likelihood, scaler


def load_gpqr_H(path=None, device=None):
    """Return GPQR model for H.

    Parameters
    ----------
    path : str or Path, optional
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

    if path is None:
        path = Path(__file__).parent / "GPQR.H.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = _load_model(CgLmcMtgpqr_H, path, device=device)
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler


def load_gpqr_phi(path=None, device=None):
    """Return GPQR model for phi.

    Parameters
    ----------
    path : str or Path, optional
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

    if path is None:
        path = Path(__file__).parent / "GPQR.phi.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = _load_model(
        CgIndependentMtgpqr_phi, path, device=device
    )
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler
