#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import os
import pathlib
import signal
import subprocess
import sys
import time

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


def nvidia_smi_devices():
    query_commands = [
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "-L"],
    ]
    for command in query_commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

        devices = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if command[-1] == "-L":
                prefix, _, _ = line.partition(":")
                _, _, index = prefix.partition(" ")
                if index:
                    devices.append(index)
            else:
                devices.append(line.split(",", 1)[0].strip())
        if devices:
            return devices

    return []


def visible_devices():
    if "HEAVYEDGE_GPU_DEVICES" in os.environ:
        override = split_device_list(os.environ.get("HEAVYEDGE_GPU_DEVICES"))
        return override if override is not None else nvidia_smi_devices()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = split_device_list(os.environ.get("CUDA_VISIBLE_DEVICES"))
        if cuda_visible_devices is None:
            return nvidia_smi_devices()
        return cuda_visible_devices

    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        nvidia_visible_devices = split_device_list(
            os.environ.get("NVIDIA_VISIBLE_DEVICES")
        )
        return (
            nvidia_visible_devices
            if nvidia_visible_devices is not None
            else nvidia_smi_devices()
        )

    return nvidia_smi_devices()


def lock_path(lock_dir, device):
    safe_prefix = "".join(c if c.isalnum() or c in "._-" else "_" for c in device)
    digest = hashlib.sha1(device.encode("utf-8")).hexdigest()[:12]
    return lock_dir / f"gpu-{safe_prefix}-{digest}.lock"


def try_lock_device(lock_dir, device):
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path(lock_dir, device), "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None

    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def acquire_device(devices, lock_dir, poll_interval, timeout):
    start = time.monotonic()
    last_notice = 0.0

    while True:
        for device in devices:
            lock_file = try_lock_device(lock_dir, device)
            if lock_file is not None:
                return device, lock_file

        elapsed = time.monotonic() - start
        if timeout is not None and elapsed >= timeout:
            raise TimeoutError(
                f"Timed out waiting for one of these GPUs: {', '.join(devices)}"
            )

        now = time.monotonic()
        if now - last_notice >= 30:
            print(
                f"[gpu-run] waiting for a free GPU among: {', '.join(devices)}",
                file=sys.stderr,
                flush=True,
            )
            last_notice = now

        time.sleep(poll_interval)


def run_child(command, env):
    child = subprocess.Popen(command, env=env)

    def forward_signal(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward_signal)

    try:
        returncode = child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a command after assigning one visible GPU with an advisory lock."
        )
    )
    parser.add_argument(
        "--lock-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("HEAVYEDGE_GPU_LOCK_DIR", "_temp/gpu-locks")
        ),
        help="Directory used to coordinate GPU assignment locks.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("HEAVYEDGE_GPU_LOCK_POLL_INTERVAL", "1")),
        help="Seconds to wait between lock attempts when all GPUs are busy.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=(
            float(os.environ["HEAVYEDGE_GPU_LOCK_TIMEOUT"])
            if "HEAVYEDGE_GPU_LOCK_TIMEOUT" in os.environ
            else None
        ),
        help="Maximum seconds to wait for a GPU. Default: wait indefinitely.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=os.environ.get("HEAVYEDGE_GPU_RUN_QUIET") == "1",
        help="Do not print GPU assignment messages.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to run")

    return args


def main():
    args = parse_args()
    devices = visible_devices()

    if not devices:
        if not args.quiet:
            print(
                "[gpu-run] no visible GPUs detected; running command unchanged",
                file=sys.stderr,
                flush=True,
            )
        return run_child(args.command, os.environ.copy())

    try:
        device, lock_file = acquire_device(
            devices,
            args.lock_dir,
            args.poll_interval,
            args.timeout,
        )
    except TimeoutError as error:
        print(f"[gpu-run] {error}", file=sys.stderr)
        return 75

    with lock_file:
        env = os.environ.copy()
        env.setdefault(
            "HEAVYEDGE_ORIGINAL_CUDA_VISIBLE_DEVICES",
            env.get("CUDA_VISIBLE_DEVICES", ""),
        )
        env["CUDA_VISIBLE_DEVICES"] = device
        env["HEAVYEDGE_ASSIGNED_CUDA_VISIBLE_DEVICES"] = device

        if not args.quiet:
            command = " ".join(args.command)
            print(
                f"[gpu-run] assigned GPU {device} to: {command}",
                file=sys.stderr,
                flush=True,
            )

        return run_child(args.command, env)


if __name__ == "__main__":
    sys.exit(main())
