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


def parse_release_version(tag):
    if not VERSION_PATTERN.fullmatch(tag):
        raise ValueError(f"Unsupported release version tag: {tag}")

    version_text = tag.removeprefix("v")
    try:
        return Version(version_text)
    except InvalidVersion as error:
        raise ValueError(f"Invalid release version tag: {tag}") from error


def trained_model_tag(tag, version):
    tag_prefix = "v" if tag.startswith("v") else ""
    pre = "" if version.pre is None else f"{version.pre[0]}{version.pre[1]}"
    return f"{tag_prefix}{version.major}.{version.minor}.{version.micro}{pre}"


def main():
    parser = argparse.ArgumentParser(description="Resolve CD workflow configuration")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    args = parser.parse_args()

    is_release = args.event_name == "release"
    is_master_push = args.event_name == "push" and args.ref_name == "master"

    # test: push/PR, release: final/pre, reuse: post/dev
    model_mode = "test"
    model_revision = ""
    model_repo_id = ""
    if is_release:
        try:
            version = parse_release_version(args.ref_name)
            if version.post is None and version.dev is None:
                model_mode = "release"
            else:
                model_mode = "reuse"
                model_revision = trained_model_tag(args.ref_name, version)
                model_repo_id = f"jeesoo9595/heavyedge-features-v{version.major}"
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)

    push_doc = int(is_release or is_master_push)

    github_output("model_mode", model_mode)
    github_output("push_doc", push_doc)
    github_output("model_revision", model_revision)
    github_output("model_repo_id", model_repo_id)


if __name__ == "__main__":
    main()
