#!/usr/bin/env python3
import argparse
import os
import re

from packaging.version import InvalidVersion, Version

VERSION_PATTERN = re.compile(
    r"^v?[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$"
)


def github_output(name, value):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main():
    parser = argparse.ArgumentParser(description="Resolve CD workflow configuration")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    args = parser.parse_args()

    is_release = args.event_name == "release"
    if is_release:
        try:
            major_version = f"v{Version(args.ref_name).major}"
            build_mode = "build"
            deploy_mode = "true"
        except InvalidVersion:
            major_version = ""
            build_mode = "build"
            deploy_mode = "false"
    else:
        major_version = ""
        build_mode = "test"
        deploy_mode = "false"

    github_output("major_version", major_version)
    github_output("build_mode", build_mode)
    github_output("deploy_mode", deploy_mode)


if __name__ == "__main__":
    main()
