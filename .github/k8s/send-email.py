#!/usr/bin/env python3

import argparse
import json
import os
import smtplib
import sys
from email.message import EmailMessage
from enum import IntEnum, IntFlag
from pathlib import Path


def env(name, default=""):
    return os.environ.get(name, default)


class BuildStatus(IntEnum):
    SUCCESS = 0
    DEPENDENCY_INSTALL_FAILED = 1
    CUDA_DEVICE_DISCOVERY_FAILED = 2
    MAKE_JOBS_DISCOVERY_FAILED = 3
    CUDA_PREFLIGHT_FAILED = 4
    BUILD_FAILED = 5


class DeployStatus(IntEnum):
    SUCCESS = 0
    MODEL_UPLOAD_FAILED = 1
    DOC_UPLOAD_FAILED = 2


class TokenStatus(IntEnum):
    SUCCESS = 0
    MISSING_ENV = 1
    PRIVATE_KEY_FILE_FAILED = 2
    JWT_SIGNING_FAILED = 3
    INSTALLATION_FETCH_FAILED = 4
    INSTALLATION_ID_PARSE_FAILED = 5
    TOKEN_CREATE_FAILED = 6
    TOKEN_PARSE_FAILED = 7


class DispatchStatus(IntEnum):
    SUCCESS = 0
    MISSING_ENV = 1
    PAYLOAD_CREATE_FAILED = 2
    REQUEST_FAILED = 3


class ContainerExitCode(IntFlag):
    SUCCESS = 0
    BUILD_FAILED = 1
    TOKEN_FAILED = 2
    DEPLOY_FAILED = 4
    DISPATCH_FAILED = 8


STATUS_DESCRIPTION_SECTIONS = {
    BuildStatus: "build",
    DeployStatus: "deploy",
    TokenStatus: "token",
    DispatchStatus: "dispatch",
}


def load_status_descriptions():
    path = Path(__file__).with_name("status-descriptions.json")
    with path.open(encoding="utf-8") as f:
        raw_descriptions = json.load(f)

    descriptions = {}
    for enum_class, section in STATUS_DESCRIPTION_SECTIONS.items():
        descriptions[enum_class] = {
            enum_class(int(key)): description
            for key, description in raw_descriptions[section].items()
        }
    return descriptions


STATUS_DESCRIPTIONS = load_status_descriptions()


def parse_status(enum_class, value):
    if value is None:
        return None
    try:
        return enum_class(value)
    except ValueError:
        return None


def status_line(label, enum_class, value):
    if value is None:
        return None
    parsed = parse_status(enum_class, value)
    if parsed is None:
        return f"{label} status: {value} UNKNOWN - Unknown status code."
    return (
        f"{label} status: {value} {parsed.name} - "
        f"{STATUS_DESCRIPTIONS[enum_class][parsed]}"
    )


def exit_code_line(exit_code):
    if exit_code is None:
        return None
    parsed = ContainerExitCode(exit_code)
    if parsed == ContainerExitCode.SUCCESS:
        return "Container exit code: 0 (success)"
    failed_phases = [
        name.removesuffix("_FAILED").lower()
        for flag in ContainerExitCode
        if flag is not ContainerExitCode.SUCCESS and flag in parsed
        for name in (flag.name,)
    ]
    known_mask = 0
    for flag in ContainerExitCode:
        known_mask |= flag.value
    unknown_bits = exit_code & ~known_mask
    details = ", ".join(failed_phases)
    if unknown_bits:
        if details:
            details = f"{details}; unknown bits: {unknown_bits}"
        else:
            details = f"unknown bits: {unknown_bits}"
    return f"Container exit code: {exit_code} ({details})"


def build_message(
    status,
    exit_code=None,
    build_status=None,
    deploy_status=None,
    token_status=None,
    dispatch_status=None,
):
    repository = env("GITHUB_REPOSITORY", "heavyedge/feature-models")
    ref_name = env("GITHUB_REF_NAME", "")
    image_tag = env("IMAGE_TAG", "")

    subject_parts = [f"[{status}]", "HeavyEdge feature-models deploy"]
    if ref_name:
        subject_parts.append(ref_name)

    body_lines = [
        f"Deployment status: {status}",
        f"Repository: {repository}",
    ]
    if ref_name:
        body_lines.append(f"Ref: {ref_name}")
    if image_tag:
        body_lines.append(f"Image tag: {image_tag}")
    exit_line = exit_code_line(exit_code)
    if exit_line:
        body_lines.append(exit_line)
    for line in (
        status_line("Build", BuildStatus, build_status),
        status_line("Deploy", DeployStatus, deploy_status),
        status_line("Token", TokenStatus, token_status),
        status_line("Dispatch", DispatchStatus, dispatch_status),
    ):
        if line:
            body_lines.append(line)

    body_lines.append(f"Dry build: {env('DRY_BUILD', 'true')}")
    body_lines.append(f"Push model: {env('PUSH_MODEL', 'false')}")
    body_lines.append(f"Push documentation: {env('PUSH_DOC', 'false')}")

    msg = EmailMessage()
    msg["From"] = env("SMTP_NOTIFY_SENDER", "heavyedge-bot@users.noreply.github.com")
    msg["To"] = env("SMTP_NOTIFY_RECIPIENT")
    msg["Subject"] = " - ".join(subject_parts)
    msg.set_content("\n".join(body_lines) + "\n")
    return msg


def main():
    parser = argparse.ArgumentParser(
        description="Send deploy status email through local SMTP relay."
    )
    parser.add_argument("--status", required=True, help="Deployment status label.")
    parser.add_argument("--exit-code", type=int, help="Build container exit code.")
    parser.add_argument("--build-status", type=int, help="Build status code.")
    parser.add_argument("--deploy-status", type=int, help="Deploy status code.")
    parser.add_argument(
        "--token-status", type=int, help="GitHub App token status code."
    )
    parser.add_argument("--dispatch-status", type=int, help="Dispatch status code.")
    parser.add_argument("--smtp-host", default=env("SMTP_HOST", "127.0.0.1"))
    parser.add_argument("--smtp-port", type=int, default=int(env("SMTP_PORT", "587")))
    args = parser.parse_args()

    if not env("SMTP_NOTIFY_RECIPIENT"):
        print("SMTP_NOTIFY_RECIPIENT is empty; skipping deployment email.")
        return 0

    message = build_message(
        args.status,
        args.exit_code,
        args.build_status,
        args.deploy_status,
        args.token_status,
        args.dispatch_status,
    )
    with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=10) as smtp:
        smtp.send_message(message)
    print(f"Sent deployment email: {args.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
