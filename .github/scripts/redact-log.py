#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser(description="Redact log file.")
parser.add_argument("input_file", type=Path)
parser.add_argument("output_file", type=Path)
parser.add_argument(
    "--secret-env",
    action="append",
    default=[],
    help="Environment-variable name whose value must be redacted. May be repeated.",
)
args = parser.parse_args()

content = args.input_file.read_text(encoding="utf-8", errors="replace")
secrets = sorted(
    {
        os.environ[name]
        for name in args.secret_env
        if os.environ.get(name) and len(os.environ[name]) >= 3
    },
    key=len,
    reverse=True,
)
for secret in secrets:
    content = content.replace(secret, "[REDACTED]")

content = re.sub(
    r"\b(?:gh[pousr]_|ghs_|github_pat_)[A-Za-z0-9_.-]+",
    "[REDACTED]",
    content,
)
content = re.sub(r"\bhf_[A-Za-z0-9_-]+\b", "[REDACTED]", content)
content = re.sub(
    r"(authorization:\s*(?:basic|bearer)\s+)[^\s]+",
    r"\1[REDACTED]",
    content,
    flags=re.IGNORECASE,
)
content = re.sub(
    r"(x-access-token:)[^\s@]+", r"\1[REDACTED]", content, flags=re.IGNORECASE
)

args.output_file.write_text(content, encoding="utf-8")
