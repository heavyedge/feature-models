#!/usr/bin/env python3

import os

from packaging.version import InvalidVersion, Version

try:
    ref_name = os.environ["GITHUB_REF_NAME"]
    major_version = f"v{Version(ref_name).major}"
except KeyError as error:
    raise SystemExit(f"Missing required environment variable: {error.args[0]}")
except InvalidVersion as error:
    raise SystemExit(f"Invalid release version {ref_name!r}: {error}")

print(major_version)
