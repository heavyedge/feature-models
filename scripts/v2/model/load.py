from pathlib import Path

import torch

from . import prior as prior_module

__all__ = [
    "load_PriorMean",
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
    return _load_prior_mean(path, device=device)
