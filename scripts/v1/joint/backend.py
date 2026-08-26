"""Device selection helpers for the joint probability scripts."""


def cuda_device(device="auto"):
    """Return ``(torch, device)`` for CUDA, or ``(None, None)`` for CPU.

    PyTorch is imported lazily so the NumPy fallback remains usable in
    lightweight environments that do not install the optional GPU stack.
    """
    if device == "cpu":
        return None, None

    try:
        import torch
    except ImportError as exc:
        if device == "auto":
            return None, None
        raise RuntimeError(
            f"device {device!r} requires PyTorch, but PyTorch is not installed"
        ) from exc

    if device == "auto":
        if not torch.cuda.is_available():
            return None, None
        return torch, torch.device("cuda")

    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError(
            "device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested but CUDA is unavailable"
        )
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {selected.index} is unavailable; "
            f"found {torch.cuda.device_count()} device(s)"
        )
    return torch, selected
