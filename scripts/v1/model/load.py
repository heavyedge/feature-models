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


def _prior_args(component, prior_name, loc_name, scale_name):
    args = dict(component["args"])
    state_dict = component["state_dict"]
    loc_key = next(
        (key for key in state_dict if key.endswith(f"{prior_name}._buffered_loc")),
        None,
    )
    scale_key = next(
        (key for key in state_dict if key.endswith(f"{prior_name}._buffered_scale")),
        None,
    )
    if loc_key is None or scale_key is None:
        args[loc_name] = None
        args[scale_name] = None
    else:
        args[loc_name] = state_dict[loc_key]
        args[scale_name] = state_dict[scale_key]
    return args


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


def _load_state_dict(module, state_dict):
    state_dict = dict(state_dict)
    expected = module.state_dict()
    compatibility_buffers = {
        "lower_counts",
        "num_quantiles",
        "quantile_level_offsets",
    }
    for key, value in expected.items():
        if key not in state_dict and key.rsplit(".", 1)[-1] in compatibility_buffers:
            state_dict[key] = value
    module.load_state_dict(state_dict)


def _load_gp(path, model_module, device, *, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    xscaler_class = getattr(scale_module, checkpoint["X_scaler"]["type"])
    X_scaler = xscaler_class(**checkpoint["X_scaler"]["args"])
    X_scaler.load_state_dict(checkpoint["X_scaler"]["state_dict"])

    yscaler_class = getattr(scale_module, checkpoint["y_scaler"]["type"])
    y_scaler = yscaler_class(**checkpoint["y_scaler"]["args"])
    y_scaler.load_state_dict(checkpoint["y_scaler"]["state_dict"])

    likelihood_class = getattr(likelihood_module, checkpoint["likelihood"]["type"])
    likelihood_args = _prior_args(
        checkpoint["likelihood"],
        "noise_prior",
        "noise_prior_loc",
        "noise_prior_scale",
    )
    likelihood = likelihood_class(**likelihood_args)
    _load_state_dict(likelihood, checkpoint["likelihood"]["state_dict"])

    model_class = getattr(model_module, checkpoint["model"]["type"])
    if model_module is gpqr_module:
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
    else:
        model_args = _prior_args(
            checkpoint["model"],
            "lengthscale_prior",
            "lengthscale_prior_loc",
            "lengthscale_prior_scale",
        )
        model_state_dict = checkpoint["model"]["state_dict"]
    model = model_class(**model_args)
    _load_state_dict(model, model_state_dict)

    X_scaler.to(device)
    y_scaler.to(device)
    likelihood.to(device)
    model.to(device)
    return checkpoint, X_scaler, y_scaler, likelihood, model


def load_GPR(path=None, device=None, *, return_metadata=False):
    """Return the independent three-batch GPR model and its transforms."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path is None:
        path = Path(__file__).parent / "gpr.pt"

    checkpoint, X_scaler, y_scaler, likelihood, model = _load_gp(
        path, gpr_module, device
    )
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

    checkpoint, X_scaler, y_scaler, likelihood, model = _load_gp(
        path, gpqr_module, device, checkpoint=checkpoint
    )
    return (
        checkpoint["quantiles"],
        float(lower_bound),
        X_scaler,
        y_scaler,
        likelihood,
        model,
    )
