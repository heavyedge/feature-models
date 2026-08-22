#!/usr/bin/env python3

import argparse
import os

from packaging.version import InvalidVersion, Version

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    choices=("major", "minor", "patch", "post", "pre", "dev", "local"),
    required=True,
    help="Version component to print.",
)
args = parser.parse_args()

try:
    ref_name = os.environ["GITHUB_REF_NAME"]
    version = Version(ref_name)
except KeyError as error:
    raise SystemExit(f"Missing required environment variable: {error.args[0]}")
except InvalidVersion as error:
    raise SystemExit(f"Invalid release version {ref_name!r}: {error}")

if args.target == "major":
    value = f"v{version.major}"
elif args.target == "minor":
    value = str(version.minor)
elif args.target == "patch":
    value = str(version.micro)
elif args.target == "pre":
    value = "" if version.pre is None else f"{version.pre[0]}{version.pre[1]}"
else:
    component = getattr(version, args.target)
    value = "" if component is None else str(component)

print(value)
