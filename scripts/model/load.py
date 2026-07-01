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
from .prior import (
    PriorMean_b2,
    PriorMean_H2,
    PriorMean_phi2,
)
from .scale import (
    MinMaxScaler,
    StandardScaler,
)

__all__ = [
    "load_H_mean",
    "load_b_mean",
    "load_phi_mean",
    "load_H_quantiles",
    "load_phi_quantiles",
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


def _load_gpr(xscaler_class, yscaler_class, mean_class, model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    X = checkpoint["train_x"]
    y = checkpoint["train_y"]

    X_scaler = xscaler_class()
    y_scaler = yscaler_class()
    mean = mean_class()
    likelihood = GaussianLikelihood()

    X_scaler.eval()
    y_scaler.eval()
    mean.eval()
    with torch.no_grad():
        X_scaled = X_scaler(X)
        res = y_scaler(y - mean(X)).squeeze(-1)
    model = model_class(
        train_x=X_scaled.detach(),
        train_y=res.detach(),
        likelihood=likelihood,
    )

    X_scaler.load_state_dict(checkpoint["x_scaler_state_dict"])
    y_scaler.load_state_dict(checkpoint["y_scaler_state_dict"])
    mean.load_state_dict(checkpoint["mean_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        mean.to(device)
        likelihood.to(device)
        model.to(device)
    return X_scaler, y_scaler, mean, likelihood, model


def load_H_mean(path=None, device=None):
    """Return GPR models for H.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    X_scaler : model_module.MinMaxScaler
    y_scaler : model_module.StandardScaler
    mean : model_module.PriorMean_H2
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_H2, GPR_H, path, device=device
    )


def load_b_mean(path=None, device=None):
    """Return GPR models for b.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    X_scaler : model_module.MinMaxScaler
    y_scaler : model_module.StandardScaler
    mean : model_module.PriorMean_b2
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "b.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_b2, GPR_b, path, device=device
    )


def load_phi_mean(path=None, device=None):
    """Return GPR models for phi.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    X_scaler : model_module.MinMaxScaler
    y_scaler : model_module.StandardScaler
    mean : model_module.PriorMean_phi2
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_phi2, GPR_phi, path, device=device
    )


def load_H_quantiles(path=None, device=None):
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
        path = Path(__file__).parent / "H.quantiles.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = _load_model(CgLmcMtgpqr_H, path, device=device)
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler


def load_phi_quantiles(path=None, device=None):
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
        path = Path(__file__).parent / "phi.quantiles.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    quantiles = checkpoint["quantiles"]

    model, likelihood, scaler = _load_model(
        CgIndependentMtgpqr_phi, path, device=device
    )
    model.to(device)
    model.eval()
    return quantiles, model, likelihood, scaler
