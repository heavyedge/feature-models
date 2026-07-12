#!/usr/bin/env python3

import argparse
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Wait for an SMTP relay to accept TCP connections."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=587)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    while True:
        try:
            with socket.create_connection(
                (args.host, args.port), timeout=args.interval
            ):
                print(f"SMTP relay is ready at {args.host}:{args.port}.")
                return 0
        except OSError:
            if time.time() >= deadline:
                print(
                    f"Timed out waiting for SMTP relay at {args.host}:{args.port}.",
                    file=sys.stderr,
                )
                return 1
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
