#!/usr/bin/env python3
import os
import smtplib
import sys
import time
from email.message import EmailMessage


def build_message(status: str) -> EmailMessage:
    recipient = os.environ["SMTP_NOTIFY_TO"]
    sender = os.environ.get("SMTP_MAIL_FROM") or "heavyedge-feature-models@localhost"
    repo = os.environ.get("GITHUB_REPOSITORY", "unknown repository")
    job_name = os.environ.get("JOB_NAME", "heavyedge-feature-models")
    git_sha = os.environ.get("GIT_SHA", "unknown")
    event_name = os.environ.get("EVENT_NAME", "unknown")
    tag_name = os.environ.get("TAG_NAME", "")
    run_url = os.environ.get("GITHUB_RUN_URL", "")

    status_labels = {
        "started": "started",
        "succeeded": "succeeded",
        "failed": "failed",
    }
    subject_status = status_labels.get(status, status)

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = f"[{subject_status}] {job_name}"

    lines = [
        f"heavyedge-feature-models {subject_status}.",
        "",
        f"Job: {job_name}",
        f"Repository: {repo}",
        f"Git SHA: {git_sha}",
        f"Event: {event_name}",
    ]
    if tag_name:
        lines.append(f"Tag: {tag_name}")
    if run_url:
        lines.append(f"GitHub Actions run: {run_url}")
    lines.extend(
        [
            "",
            f"HEAVYEDGE_TEST_MODE={os.environ.get('HEAVYEDGE_TEST_MODE', '')}",
            f"UPLOAD_TO_HUGGINGFACE={os.environ.get('UPLOAD_TO_HUGGINGFACE', '')}",
            f"PUSH_DOC={os.environ.get('PUSH_DOC', '')}",
        ]
    )
    message.set_content("\n".join(lines))
    return message


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: notify-deploy.py <started|succeeded|failed>", file=sys.stderr)
        return 2

    status = sys.argv[1]
    recipient = os.environ["SMTP_NOTIFY_TO"]
    host = os.environ.get("SMTP_HOST", "127.0.0.1")
    port = int(os.environ.get("SMTP_PORT", "587"))
    retries = int(os.environ.get("SMTP_NOTIFY_RETRIES", "12"))
    delay = float(os.environ.get("SMTP_NOTIFY_RETRY_DELAY", "5"))
    message = build_message(status)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.send_message(message)
            print(f"Sent {status} notification to {recipient}.")
            return 0
        except Exception as exc:
            last_error = exc
            print(
                f"Notification attempt {attempt}/{retries} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < retries:
                time.sleep(delay)

    print(f"Failed to send {status} notification: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
