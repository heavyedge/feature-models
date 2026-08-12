#!/usr/bin/env python3

import os

from packaging.version import InvalidVersion, Version

try:
    ref_name = os.environ["GITHUB_REF_NAME"]
    major_version = f"v{Version(ref_name).major}"
except InvalidVersion:
    major_version = ""

print(major_version)
