#!/usr/bin/env python3
import argparse
import os
import re
import sys

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    from setuptools._vendor.packaging.version import InvalidVersion, Version


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


def trained_model_can_be_released(tag):
    if not VERSION_PATTERN.fullmatch(tag):
        print(f"Unsupported release version tag: {tag}", file=sys.stderr)
        return False

    version_text = tag.removeprefix("v")
    try:
        version = Version(version_text)
    except InvalidVersion:
        print(f"Invalid release version tag: {tag}", file=sys.stderr)
        return False

    return version.post is None


def main():
    parser = argparse.ArgumentParser(description="Resolve CD workflow flags")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    args = parser.parse_args()

    is_release = args.event_name == "release"
    upload_to_huggingface = int(
        is_release and trained_model_can_be_released(args.ref_name)
    )
    push_doc = int(is_release)

    github_output("upload_to_huggingface", upload_to_huggingface)
    github_output("push_doc", push_doc)


if __name__ == "__main__":
    main()
