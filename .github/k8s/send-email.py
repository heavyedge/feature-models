#!/usr/bin/env python3

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage


def env(name, default=""):
    return os.environ.get(name, default)


def build_message(status):
    repository = env("GITHUB_REPOSITORY", "heavyedge/feature-models")
    ref_name = env("GITHUB_REF_NAME", "")
    image_tag = env("IMAGE_TAG", "")

    subject_parts = ["HeavyEdge feature-models deploy", status]
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
    if env("UPLOAD_TO_HUGGINGFACE"):
        body_lines.append(f"Upload to Hugging Face: {env('UPLOAD_TO_HUGGINGFACE')}")
    if env("PUSH_DOC"):
        body_lines.append(f"Push documentation: {env('PUSH_DOC')}")

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
    parser.add_argument("--smtp-host", default=env("SMTP_HOST", "127.0.0.1"))
    parser.add_argument("--smtp-port", type=int, default=int(env("SMTP_PORT", "587")))
    args = parser.parse_args()

    if not env("SMTP_NOTIFY_RECIPIENT"):
        print("SMTP_NOTIFY_RECIPIENT is empty; skipping deployment email.")
        return 0

    message = build_message(args.status)
    with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=10) as smtp:
        smtp.send_message(message)
    print(f"Sent deployment email: {args.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
