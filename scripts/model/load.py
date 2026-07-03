from pathlib import Path

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch_qr.likelihoods import CenterGapQuantileLikelihood

from .gpqr import (
    CenterGapMTGPQR_H,
    CenterGapMTGPQR_phi,
)
from .gpr import (
    GPR_H,
    GPR_b,
    GPR_phi,
)
from .prior import (
    PriorMean_b,
    PriorMean_H,
    PriorMean_phi,
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


def _load_gpr(xscaler_class, yscaler_class, mean_class, model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    X = checkpoint["train_x"]
    y = checkpoint["train_y"]
    dim = X.shape[-1]
    batch_shape = X.shape[:-2]

    X_scaler = xscaler_class(dim, batch_shape=batch_shape)
    y_scaler = yscaler_class(1, batch_shape=batch_shape)
    mean = mean_class(batch_shape=batch_shape)
    likelihood = GaussianLikelihood(batch_shape=batch_shape)

    X_scaler.load_state_dict(checkpoint["X_scaler_state_dict"])
    y_scaler.load_state_dict(checkpoint["y_scaler_state_dict"])
    mean.load_state_dict(checkpoint["mean_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

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
    mean : model_module.PriorMean_H
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_H, GPR_H, path, device=device
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
    mean : model_module.PriorMean_b
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "b.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_b, GPR_b, path, device=device
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
    mean : model_module.PriorMean_phi
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    model : gpytorch.models.ExactGP
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.mean.pt"
    return _load_gpr(
        MinMaxScaler, StandardScaler, PriorMean_phi, GPR_phi, path, device=device
    )


def _load_gpqr(
    xscaler_class, yscaler_class, mean_class, model_class, path, device=None
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    X = checkpoint["train_x"]
    dim = X.shape[-1]
    batch_shape = X.shape[:-2]
    inducing_points = checkpoint["inducing_points"]
    quantiles = checkpoint["quantiles"]
    num_lower_quantiles = checkpoint["num_lower_quantiles"]
    num_latents = checkpoint["num_latents"]

    X_scaler = xscaler_class(dim, batch_shape=batch_shape)
    y_scaler = yscaler_class(1, batch_shape=batch_shape)
    mean = mean_class(batch_shape=batch_shape)
    likelihood = CenterGapQuantileLikelihood(
        quantiles.unsqueeze(0),
        num_lower_quantiles,
        torch.zeros((*batch_shape, len(quantiles))),
        learn_scales=True,
    ).to(device)
    model = model_class(
        inducing_points=inducing_points,
        num_quantiles=len(quantiles),
        num_lower_quantiles=num_lower_quantiles,
        num_latents=num_latents,
        batch_shape=batch_shape,
    )

    X_scaler.load_state_dict(checkpoint["X_scaler_state_dict"])
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
    return quantiles, X_scaler, y_scaler, mean, likelihood, model


def load_H_quantiles(path=None, device=None):
    """Return GPQR model for H.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    quantiles
    X_scaler
    y_scaler
    mean
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.quantiles.pt"
    return _load_gpqr(
        MinMaxScaler,
        StandardScaler,
        PriorMean_H,
        CenterGapMTGPQR_H,
        path,
        device=device,
    )


def load_phi_quantiles(path=None, device=None):
    """Return GPQR model for phi.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    quantiles
    X_scaler
    y_scaler
    mean
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.quantiles.pt"
    return _load_gpqr(
        MinMaxScaler,
        StandardScaler,
        PriorMean_phi,
        CenterGapMTGPQR_phi,
        path,
        device=device,
    )
