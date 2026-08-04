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


def parse_tag(tag, version):
    tag_prefix = "v" if tag.startswith("v") else ""
    pre = "" if version.pre is None else f"{version.pre[0]}{version.pre[1]}"
    return f"{tag_prefix}{version.major}.{version.minor}.{version.micro}{pre}"


def main():
    parser = argparse.ArgumentParser(description="Resolve CD workflow configuration")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-name", required=True)
    args = parser.parse_args()

    is_release = args.event_name == "release"

    build_mode = "test"  # build: final/pre, pull: post, test: dev/push&PR
    deploy_mode = "false"  # true: final/pre/post, false: post/dev/push&PR
    doc_build_mode = "test"  # build: final/pre/post, test: dev/push&PR
    doc_deploy_mode = "false"  # true: final/pre/post, false: dev/push&PR
    upstream_revision = ""
    upstream_repo_id = ""
    if is_release:
        try:
            version = parse_release_version(args.ref_name)
            if version.dev is not None:
                pass
            elif version.post is not None:
                build_mode = "pull"
                upstream_revision = parse_tag(args.ref_name, version)
                upstream_repo_id = f"heavyedge/feature-model-v{version.major}"
                deploy_mode = "true"
                doc_build_mode = "build"
                doc_deploy_mode = "true"
            else:
                build_mode = "build"
                deploy_mode = "true"
                doc_build_mode = "build"
                doc_deploy_mode = "true"
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)

    github_output("build_mode", build_mode)
    github_output("deploy_mode", deploy_mode)
    github_output("doc_build_mode", doc_build_mode)
    github_output("doc_deploy_mode", doc_deploy_mode)
    github_output("upstream_revision", upstream_revision)
    github_output("upstream_repo_id", upstream_repo_id)


if __name__ == "__main__":
    main()
