#!/usr/bin/env python3

import argparse
import os
import smtplib
import socket
import sys
import time
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


def send_with_retry(message, host, port, attempts, delay):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.send_message(message)
            return
        except (OSError, smtplib.SMTPException, socket.timeout) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(delay)

    raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Send deploy status email through local SMTP relay."
    )
    parser.add_argument("--status", required=True, help="Deployment status label.")
    parser.add_argument("--smtp-host", default=env("SMTP_HOST", "127.0.0.1"))
    parser.add_argument("--smtp-port", type=int, default=int(env("SMTP_PORT", "587")))
    parser.add_argument("--attempts", type=int, default=int(env("SMTP_ATTEMPTS", "12")))
    parser.add_argument(
        "--delay", type=float, default=float(env("SMTP_RETRY_DELAY", "5"))
    )
    args = parser.parse_args()

    if not env("SMTP_NOTIFY_RECIPIENT"):
        print("SMTP_NOTIFY_RECIPIENT is empty; skipping deployment email.")
        return 0

    message = build_message(args.status)
    send_with_retry(message, args.smtp_host, args.smtp_port, args.attempts, args.delay)
    print(f"Sent deployment email: {args.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
