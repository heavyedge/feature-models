#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    from setuptools._vendor.packaging.version import InvalidVersion, Version


IGNORED_DIRS = {".git", ".github", "_static"}
WHITELISTED_VERSION_DIRS = {"latest", "stable"}


def parse_version_dir(path):
    if not path.is_dir() or path.name in IGNORED_DIRS:
        return None

    try:
        return Version(path.name)
    except InvalidVersion:
        return None


def stable_version(versions):
    stable_versions = [
        (version, name)
        for version, name in versions
        if not version.is_prerelease and not version.is_devrelease
    ]
    if stable_versions:
        return max(stable_versions)[1]

    return None


def version_names(site_dir, versions):
    names = [name for _, name in versions]
    whitelisted_names = sorted(
        path.name
        for path in site_dir.iterdir()
        if path.is_dir() and path.name in WHITELISTED_VERSION_DIRS
    )

    return names + [name for name in whitelisted_names if name not in names]


def write_stable_redirect(site_dir):
    stable_dir = site_dir / "stable"
    shutil.rmtree(stable_dir, ignore_errors=True)
    stable_dir.mkdir(parents=True)
    stable_dir.joinpath("index.html").write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>HeavyEdge Feature Models stable documentation</title>
  </head>
  <body>
    <p id="message">Redirecting to the stable documentation.</p>
    <script>
      fetch("../versions.json", { cache: "no-store" })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Unable to load documentation versions.");
          }
          return response.json();
        })
        .then((metadata) => {
          if (!metadata.stable) {
            throw new Error("No stable documentation version is available.");
          }
          window.location.replace(`../${metadata.stable}/`);
        })
        .catch((error) => {
          document.getElementById("message").textContent = error.message;
        });
    </script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", required=True)
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    versions = sorted(
        (
            (version, path.name)
            for path in site_dir.iterdir()
            if (version := parse_version_dir(path)) is not None
        ),
        key=lambda item: item[0],
    )
    stable = stable_version(versions)
    write_stable_redirect(site_dir)
    names = version_names(site_dir, versions)

    metadata = {
        "stable": stable,
        "versions": names,
    }
    (site_dir / "versions.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Stable documentation version: {stable or '(none)'}")
    print(f"Documented versions: {', '.join(names) or '(none)'}")


if __name__ == "__main__":
    main()
