#!/usr/bin/env python3
import argparse
import json
import re
import shutil
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
    release_versions = sorted(
        (
            path.name
            for path in site_dir.iterdir()
            if path.is_dir() and VERSION_PATTERN.fullmatch(path.name)
        ),
        key=version_key,
    )

    stable = release_versions[-1] if release_versions else None
    has_latest = (site_dir / "latest").is_dir()
    versions = []
    if has_latest:
        versions.append("latest")
    if stable:
        versions.append("stable")
    versions.extend(release_versions)

    if stable:
        stable_dir = site_dir / "stable"
        shutil.rmtree(stable_dir, ignore_errors=True)
        stable_dir.mkdir(parents=True)
        stable_dir.joinpath("index.html").write_text(
            f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=../{stable}/">
    <title>HeavyEdge Feature Models stable documentation</title>
  </head>
  <body>
    <p><a href="../{stable}/">Stable documentation</a></p>
  </body>
</html>
""",
            encoding="utf-8",
        )

    latest = "latest" if has_latest else stable or args.fallback_version

    metadata = {
        "latest": latest,
        "stable": stable,
        "releases": release_versions,
        "versions": versions,
    }
    (site_dir / "versions.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Latest documentation URL resolves to: {latest}.")
    print(f"Stable documentation URL resolves to: {stable}.")


if __name__ == "__main__":
    main()
