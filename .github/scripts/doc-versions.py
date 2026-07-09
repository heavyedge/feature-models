#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

VERSION_PATTERN = re.compile(r"^v?[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$")


def version_key(version):
    normalized = version[1:] if version.startswith("v") else version
    base, _, post = normalized.partition(".post")
    numbers = tuple(int(part) for part in base.split("."))
    post_number = int(post) if post else -1
    return (*numbers, post_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", required=True)
    parser.add_argument("--fallback-version", required=True)
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    versions = sorted(
        (
            path.name
            for path in site_dir.iterdir()
            if path.is_dir() and VERSION_PATTERN.fullmatch(path.name)
        ),
        key=version_key,
    )

    if not versions:
        versions = [args.fallback_version]

    metadata = {
        "latest": versions[-1],
        "versions": versions,
    }
    (site_dir / "versions.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    v = versions[-1]
    print(f"Root documentation URL resolves latest version from versions.json: {v}.")


if __name__ == "__main__":
    main()
