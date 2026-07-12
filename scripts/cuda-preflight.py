#!/usr/bin/env python3
import argparse
import os
import sys

import torch

NO_GPU_VALUES = {"", "-1", "none", "void", "nodevfiles"}


def split_device_list(value):
    if value is None:
        return None

    value = value.strip()
    if value.lower() in NO_GPU_VALUES:
        return []
    if value.lower() == "all":
        return None

    return [device.strip() for device in value.split(",") if device.strip()]


def environment_devices():
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if name in os.environ:
            return split_device_list(os.environ.get(name))
    return None


def cuda_devices():
    if not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    devices = environment_devices()
    if devices is not None and len(devices) == count:
        return devices

    return [str(index) for index in range(count)]


def configured_gpu_count():
    devices = os.environ.get("HEAVYEDGE_GPU_DEVICES", "")
    return len([device for device in devices.split(",") if device.strip()])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-devices",
        action="store_true",
        help="Print accessible CUDA device indexes as a comma-separated list.",
    )
    parser.add_argument(
        "--print-count",
        action="store_true",
        help="Print the number of accessible CUDA devices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    devices = cuda_devices()
    if args.print_devices:
        print(",".join(devices))
        return 0 if devices else 1
    if args.print_count:
        print(len(devices))
        return 0 if devices else 1

    expected = configured_gpu_count()
    available = torch.cuda.is_available()
    count = len(devices)

    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("NVIDIA_VISIBLE_DEVICES=", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    print("HEAVYEDGE_GPU_DEVICES=", os.environ.get("HEAVYEDGE_GPU_DEVICES"))
    print("torch.cuda.is_available=", available)
    print("torch.cuda.device_count=", count)
    print("accessible CUDA devices=", ",".join(devices))

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
