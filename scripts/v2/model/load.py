from pathlib import Path

import torch

from . import gpqr as gpqr_module
from . import gpr as gpr_module
from . import likelihoods as likelihood_module
from . import prior as prior_module
from . import scale as scale_module

__all__ = [
    "load_PriorMean",
    "load_GPR",
    "load_GPQR",
]


def load_PriorMean(path=None, device=None):
    """Return prior mean model.

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
        path = Path(__file__).parent / "prior_mean.pt"

    checkpoint = torch.load(path, map_location=device, weights_only=False)["model"]
    model_class = getattr(prior_module, checkpoint["type"])
    args = checkpoint["args"]
    model = model_class(batch_shape=args["batch_shape"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model


def load_GPR(path=None, device=None):
    """Return GPR model.

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
        path = Path(__file__).parent / "gpr.pt"

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    xscaler_class = getattr(scale_module, checkpoint["X_scaler"]["type"])
    X_scaler = xscaler_class(**checkpoint["X_scaler"]["args"])
    X_scaler.load_state_dict(checkpoint["X_scaler"]["state_dict"])

    yscaler_class = getattr(scale_module, checkpoint["y_scaler"]["type"])
    y_scaler = yscaler_class(**checkpoint["y_scaler"]["args"])
    y_scaler.load_state_dict(checkpoint["y_scaler"]["state_dict"])

    likelihood_class = getattr(likelihood_module, checkpoint["likelihood"]["type"])
    likelihood = likelihood_class(**checkpoint["likelihood"]["args"])
    likelihood.load_state_dict(checkpoint["likelihood"]["state_dict"])

    model_class = getattr(gpr_module, checkpoint["model"]["type"])
    model = model_class(**checkpoint["model"]["args"])
    model.load_state_dict(checkpoint["model"]["state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        likelihood.to(device)
        model.to(device)

    return X_scaler, y_scaler, likelihood, model


def load_GPQR(path=None, device=None):
    """Return GPQR model.

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
        path = Path(__file__).parent / "gpqr.pt"

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    quantiles = checkpoint["quantiles"]

    xscaler_class = getattr(scale_module, checkpoint["X_scaler"]["type"])
    X_scaler = xscaler_class(**checkpoint["X_scaler"]["args"])
    X_scaler.load_state_dict(checkpoint["X_scaler"]["state_dict"])

    yscaler_class = getattr(scale_module, checkpoint["y_scaler"]["type"])
    y_scaler = yscaler_class(**checkpoint["y_scaler"]["args"])
    y_scaler.load_state_dict(checkpoint["y_scaler"]["state_dict"])

    likelihood_class = getattr(likelihood_module, checkpoint["likelihood"]["type"])
    likelihood = likelihood_class(**checkpoint["likelihood"]["args"])
    likelihood.load_state_dict(checkpoint["likelihood"]["state_dict"])

    model_class = getattr(gpqr_module, checkpoint["model"]["type"])
    model = model_class(**checkpoint["model"]["args"])
    model.load_state_dict(checkpoint["model"]["state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        likelihood.to(device)
        model.to(device)

    return quantiles, X_scaler, y_scaler, likelihood, model
