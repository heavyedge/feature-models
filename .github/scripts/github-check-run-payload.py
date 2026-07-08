#!/usr/bin/env python3

import datetime
import json
import sys


def utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "Usage: github-check-run-payload.py <started|succeeded|failed> "
            "<create|update> <check-name> <git-sha> <job-name> <github-run-url>",
            file=sys.stderr,
        )
        return 2

    action, payload_kind, check_name, git_sha, job_name, github_run_url = sys.argv[1:7]
    if action not in {"started", "succeeded", "failed"}:
        print(f"Invalid action: {action}", file=sys.stderr)
        return 2
    if payload_kind not in {"create", "update"}:
        print(f"Invalid payload kind: {payload_kind}", file=sys.stderr)
        return 2

    payload = {
        "details_url": github_run_url,
        "output": {
            "title": check_name,
            "summary": (
                "Kubernetes GPU job started."
                if action == "started"
                else "Kubernetes GPU job completed."
            ),
        },
    }

    now = utc_now()
    if action == "started":
        payload["status"] = "in_progress"
        payload["started_at"] = now
    elif action == "succeeded":
        payload["status"] = "completed"
        payload["conclusion"] = "success"
        payload["completed_at"] = now
        payload["output"]["summary"] = "Kubernetes GPU job succeeded."
    else:
        payload["status"] = "completed"
        payload["conclusion"] = "failure"
        payload["completed_at"] = now
        payload["output"]["summary"] = "Kubernetes GPU job failed."

    if payload_kind == "create":
        payload["name"] = check_name
        payload["head_sha"] = git_sha
        payload["external_id"] = job_name

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
