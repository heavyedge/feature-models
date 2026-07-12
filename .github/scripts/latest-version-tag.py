#!/usr/bin/env python3
import subprocess

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    from setuptools._vendor.packaging.version import InvalidVersion, Version


def main():
    versions = []
    for tag in subprocess.check_output(
        ["git", "tag", "--merged", "HEAD", "--list"],
        text=True,
    ).splitlines():
        try:
            version = Version(tag.removeprefix("v"))
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        versions.append((version, tag))

    if versions:
        print(max(versions)[1])


if __name__ == "__main__":
    main()
