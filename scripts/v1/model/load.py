from pathlib import Path

import torch

from .prior import (
    PriorMean_H,
    PriorMean_phi,
)

__all__ = [
    "load_PriorMean_H",
    "load_PriorMean_phi",
]


def _load_prior_mean(model_class, path, device=None):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model = model_class().to(device)
    model.load_state_dict(state_dict)
    return model


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
