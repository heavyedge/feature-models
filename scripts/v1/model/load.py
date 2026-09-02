from pathlib import Path

import torch
import torch.nn.functional as F

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
    """Return the batched prior-mean model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path is None:
        path = Path(__file__).parent / "prior_mean.pt"

    checkpoint = torch.load(path, map_location=device, weights_only=False)["model"]
    model_class = getattr(prior_module, checkpoint["type"])
    model = model_class(batch_shape=checkpoint["args"]["batch_shape"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def load_GPR(path=None, device=None, *, return_metadata=False):
    """Return the independent three-batch GPR model and its transforms."""
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

    X_scaler.to(device)
    y_scaler.to(device)
    likelihood.to(device)
    model.to(device)
    result = (X_scaler, y_scaler, likelihood, model)
    if return_metadata:
        return (*result, checkpoint.get("metadata", {}))
    return result


def load_GPQR(path=None, device=None):
    """Return the independent three-batch GPQR model and its transforms."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path is None:
        path = Path(__file__).parent / "gpqr.pt"

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    legacy_model_lower_bound = checkpoint["model"]["args"].pop(
        "quantile_slope_lower_bound", None
    )
    legacy_likelihood_lower_bound = checkpoint["likelihood"]["args"].pop(
        "quantile_slope_lower_bound", None
    )
    lower_bound = checkpoint.get(
        "quantile_gap_lower_bound",
        legacy_model_lower_bound or legacy_likelihood_lower_bound or 1e-4,
    )
    checkpoint["model"]["args"]["quantile_levels"] = checkpoint["quantiles"]

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
    model_args = dict(checkpoint["model"]["args"])
    # Checkpoints written before GPQR lengthscales became mandatory stored
    # either ``fixed_lengthscale`` or a learned raw lengthscale.  Preserve
    # their fitted value while loading them under the new fixed-only API.
    lengthscale = model_args.pop("lengthscale", None)
    if lengthscale is None:
        lengthscale = model_args.pop("fixed_lengthscale", None)
    if lengthscale is None:
        raw_lengthscale = checkpoint["model"]["state_dict"].get(
            "covar_module.base_kernel.raw_lengthscale"
        )
        if raw_lengthscale is None:
            raise ValueError("GPQR checkpoint is missing its lengthscale")
        lengthscale = F.softplus(raw_lengthscale)
    model_args["lengthscale"] = lengthscale
    model_args.pop("lengthscale_prior_loc", None)
    model_args.pop("lengthscale_prior_scale", None)
    model_state_dict = {
        key: value
        for key, value in checkpoint["model"]["state_dict"].items()
        if "lengthscale_prior" not in key
    }
    model = model_class(**model_args)
    model.load_state_dict(model_state_dict)

    X_scaler.to(device)
    y_scaler.to(device)
    likelihood.to(device)
    model.to(device)
    return (
        checkpoint["quantiles"],
        float(lower_bound),
        X_scaler,
        y_scaler,
        likelihood,
        model,
    )
