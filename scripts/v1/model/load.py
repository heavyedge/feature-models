from pathlib import Path

import torch
from gpytorch.constraints import Positive
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior

from .gpr import (
    GPR_H,
    GPR_phi,
)
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
]


def _load_prior_mean(model_class, path, device=None):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model = model_class().to(device)
    model.load_state_dict(state_dict)
    return model


def _load_gpr(xscaler_class, yscaler_class, mean_class, model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    X = checkpoint["train_x"]
    y = checkpoint["train_y"]
    dim = X.shape[-1]
    batch_shape = X.shape[:-2]

    X_scaler = xscaler_class(dim, batch_shape=batch_shape)
    y_scaler = yscaler_class(1, batch_shape=batch_shape)
    mean = mean_class(batch_shape=batch_shape)
    likelihood = GaussianLikelihood(
        batch_shape=batch_shape,
        noise_prior=LogNormalPrior(0, 1),
        noise_constraint=Positive(),
    )

    X_scaler.load_state_dict(checkpoint["X_scaler_state_dict"])
    y_scaler.load_state_dict(checkpoint["y_scaler_state_dict"])
    mean.load_state_dict(checkpoint["mean_state_dict"])

    if device is not None:
        X_scaler.to(device)
        y_scaler.to(device)
        mean.to(device)
        likelihood.to(device)

    X_scaler.eval()
    y_scaler.eval()
    mean.eval()
    X_scaled = X_scaler(X)
    residual = y_scaler((y - mean(X)).unsqueeze(-1)).squeeze(-1)
    model = model_class(X_scaled, residual, likelihood, batch_shape=batch_shape)

    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model.to(device)
    return X_scaler, y_scaler, mean, likelihood, model


def load_PriorMean_H(path=None, device=None):
    """Return prior mean model for H.

    Parameters
    ----------
    path : str or Path, optional
    device : torch.device, optional
        Device to run the model on. If None, uses CUDA if available, else CPU.
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
    mean
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
        PriorMean_H,
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
    mean
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
        PriorMean_phi,
        GPR_phi,
        path,
        device=device,
    )
