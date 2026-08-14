from pathlib import Path

import torch

from . import gpqr as gpqr_module
from . import gpr as gpr_module
from . import likelihoods as likelihood_module
from . import prior as prior_module
from . import scale as scale_module

__all__ = [
    "load_PriorMean_H",
    "load_PriorMean_phi",
    "load_GPR_H",
    "load_GPR_phi",
    "load_GPQR_H",
    "load_GPQR_phi",
]


def _load_prior_mean(path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)["model"]
    model_class = getattr(prior_module, checkpoint["type"])
    args = checkpoint["args"]
    model = model_class(batch_shape=args["batch_shape"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model


def _load_gpr(path, device=None):
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


def _load_gpqr(path, device=None):
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
    return _load_prior_mean(path, device=device)


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
    return _load_prior_mean(path, device=device)


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
    return _load_gpr(path, device=device)


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
    return _load_gpr(path, device=device)


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
    return _load_gpqr(path, device=device)


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
    return _load_gpqr(path, device=device)
