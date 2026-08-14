from pathlib import Path

import torch

from .gpqr import (
    CenterGapMTGPQR_H,
    CenterGapMTGPQR_phi,
)
from .gpr import (
    GPR_H,
    GPR_phi,
)
from .likelihoods import CenterGapQuantilesLikelihood, GaussianLikelihood
from .prior import (
    PriorMean_H,
    PriorMean_phi,
)
from .scale import (
    MinMaxScaler,
    StandardScaler,
)

__all__ = [
    "load_PriorMean_H",
    "load_PriorMean_phi",
    "load_GPR_H",
    "load_GPR_phi",
    "load_GPQR_H",
    "load_GPQR_phi",
]


def _load_prior_mean(model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)["model"]
    args = checkpoint["args"]
    model = model_class(batch_shape=args["batch_shape"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model


def _load_gpr(xscaler_class, yscaler_class, model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    X_scaler = xscaler_class(**checkpoint["X_scaler"]["args"])
    X_scaler.load_state_dict(checkpoint["X_scaler"]["state_dict"])

    y_scaler = yscaler_class(**checkpoint["y_scaler"]["args"])
    y_scaler.load_state_dict(checkpoint["y_scaler"]["state_dict"])

    likelihood = GaussianLikelihood(**checkpoint["likelihood"]["args"])
    likelihood.load_state_dict(checkpoint["likelihood"]["state_dict"])

    model_args = checkpoint["model"]["args"]
    model_args.update(likelihood=likelihood)
    model = model_class(**model_args)
    model.load_state_dict(checkpoint["model"]["state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        likelihood.to(device)
        model.to(device)

    return X_scaler, y_scaler, likelihood, model


def _load_gpqr(
    xscaler_class, yscaler_class, mean_class, model_class, path, device=None
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    quantiles = checkpoint["quantiles"]

    X_scaler = xscaler_class(**checkpoint["X_scaler"]["args"])
    X_scaler.load_state_dict(checkpoint["X_scaler"]["state_dict"])

    y_scaler = yscaler_class(**checkpoint["y_scaler"]["args"])
    y_scaler.load_state_dict(checkpoint["y_scaler"]["state_dict"])

    likelihood = CenterGapQuantilesLikelihood(**checkpoint["likelihood"]["args"])
    likelihood.load_state_dict(checkpoint["likelihood"]["state_dict"])

    model = model_class(**checkpoint["model"]["args"])
    model.load_state_dict(checkpoint["model"]["state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        likelihood.to(device)
        model.to(device)

    return quantiles, X_scaler, y_scaler, likelihood, model


def load_PriorMean_H(path=None, device=None):
    """Return prior mean model for H.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.prior_mean.pt"
    return _load_prior_mean(PriorMean_H, path, device=device)


def load_PriorMean_phi(path=None, device=None):
    """Return prior mean model for phi.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.prior_mean.pt"
    return _load_prior_mean(PriorMean_phi, path, device=device)


def load_GPR_H(path=None, device=None):
    """Return GPR model for H.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    X_scaler
    y_scaler
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.gpr.pt"
    return _load_gpr(
        MinMaxScaler,
        StandardScaler,
        GPR_H,
        path,
        device=device,
    )


def load_GPR_phi(path=None, device=None):
    """Return GPR model for phi.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.

    Returns
    -------
    X_scaler
    y_scaler
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.gpr.pt"
    return _load_gpr(
        MinMaxScaler,
        StandardScaler,
        GPR_phi,
        path,
        device=device,
    )


def load_GPQR_H(path=None, device=None):
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
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "H.gpqr.pt"
    return _load_gpqr(
        MinMaxScaler,
        StandardScaler,
        PriorMean_H,
        CenterGapMTGPQR_H,
        path,
        device=device,
    )


def load_GPQR_phi(path=None, device=None):
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
    likelihood
    model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        path = Path(__file__).parent / "phi.gpqr.pt"
    return _load_gpqr(
        MinMaxScaler,
        StandardScaler,
        PriorMean_phi,
        CenterGapMTGPQR_phi,
        path,
        device=device,
    )
