#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
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


def trained_model_can_be_released(tag):
    version = parse_release_version(tag)
    return version.post is None and version.dev is None


def trained_model_tag(tag):
    version = parse_release_version(tag)
    tag_prefix = "v" if tag.startswith("v") else ""
    pre = "" if version.pre is None else f"{version.pre[0]}{version.pre[1]}"
    return f"{tag_prefix}{version.major}.{version.minor}.{version.micro}{pre}"


def latest_trained_model_tag():
    versions = []
    for tag in subprocess.check_output(
        ["git", "tag", "--merged", "HEAD", "--list"],
        text=True,
    ).splitlines():
        if not VERSION_PATTERN.fullmatch(tag):
            continue
        version = parse_release_version(tag)
        if version.post is not None or version.dev is not None:
            continue
        versions.append((version, tag))

    if not versions:
        raise ValueError("No final or pre-release trained model tag is reachable from HEAD")

    return max(versions)[1]


def model_repo_id(tag):
    version = parse_release_version(tag)
    return f"jeesoo9595/heavyedge-features-v{version.major}"


def main():
    parser = argparse.ArgumentParser(description="Resolve CD workflow flags")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    args = parser.parse_args()

    is_release = args.event_name == "release"
    is_master_push = args.event_name == "push" and args.ref_name == "master"

    try:
        model_revision = (
            trained_model_tag(args.ref_name)
            if is_release
            else latest_trained_model_tag()
        )
        push_model = int(
            is_release and trained_model_can_be_released(args.ref_name)
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    # Every repository release publishes a base image. The inference image is
    # filtered later using push_model, because it is tied to a new trained model.
    pull_model = int(is_release and not push_model)
    push_image = int(is_release)
    push_doc = int(is_release or is_master_push)

    github_output("push_model", push_model)
    github_output("pull_model", pull_model)
    github_output("push_image", push_image)
    github_output("push_doc", push_doc)
    github_output("model_revision", model_revision)
    github_output("model_repo_id", model_repo_id(model_revision))


if __name__ == "__main__":
    main()
