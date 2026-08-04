#!/usr/bin/env python3

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from enum import IntEnum, IntFlag


def env(name, default=""):
    return os.environ.get(name, default)


class BuildStatus(IntEnum):
    SUCCESS = 0
    SETUP_FAILED = 1
    BUILD_FAILED = 2
    TEST_FAILED = 3
    DOC_BUILD_FAILED = 4


class DeployStatus(IntFlag):
    SUCCESS = 0
    SKIPPED = 1
    DEPLOY_FAILED = 2
    DOC_DEPLOY_FAILED = 4


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
    if issubclass(enum_class, IntFlag) and parsed != enum_class.SUCCESS:
        flags = [
            flag for flag in enum_class if flag != enum_class.SUCCESS and flag in parsed
        ]
        names = ", ".join(flag.name for flag in flags)
        return f"{label} status: {value} {names}"
    return f"{label} status: {value} {parsed.name}"


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
    repository = env("GITHUB_REPOSITORY")
    ref_name = env("GITHUB_REF_NAME")

    job_name = env("KUBERNETES_JOB_NAME")

    body_lines = [
        f"Deployment status: {status}",
        f"Repository: https://github.com/{repository}",
    ]
    if ref_name:
        body_lines.append(f"Ref: {ref_name}")
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

    body_lines.append(f"BUILD_MODE: {env('BUILD_MODE')}")
    body_lines.append(f"DEPLOY_MODE: {env('DEPLOY_MODE')}")
    body_lines.append(f"DOC_BUILD_MODE: {env('DOC_BUILD_MODE')}")
    body_lines.append(f"DOC_DEPLOY_MODE: {env('DOC_DEPLOY_MODE')}")

    msg = EmailMessage()
    msg["To"] = env("SMTP_NOTIFY_RECIPIENT")
    msg["Subject"] = f"[{status}] {job_name}"
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
