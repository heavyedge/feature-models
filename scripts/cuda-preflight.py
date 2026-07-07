#!/usr/bin/env python3
import os
import sys

import torch


def configured_gpu_count():
    devices = os.environ.get("HEAVYEDGE_GPU_DEVICES", "")
    return len([device for device in devices.split(",") if device.strip()])


def main():
    expected = configured_gpu_count()
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()

    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("NVIDIA_VISIBLE_DEVICES=", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    print("HEAVYEDGE_GPU_DEVICES=", os.environ.get("HEAVYEDGE_GPU_DEVICES"))
    print("torch.cuda.is_available=", available)
    print("torch.cuda.device_count=", count)

    if not available:
        print("CUDA GPUs are not visible to PyTorch.", file=sys.stderr)
        return 1
    if expected and count < expected:
        print(
            f"Expected at least {expected} CUDA GPUs, but PyTorch sees {count}.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
