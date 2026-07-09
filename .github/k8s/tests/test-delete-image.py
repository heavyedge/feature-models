#!/usr/bin/env python3
import contextlib
import http.server
import os
import subprocess
import sys
import threading

SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "delete-image.sh")
)


class RegistryHandler(http.server.BaseHTTPRequestHandler):
    scenario = {}
    requests = []

    def log_message(self, _format, *_args):
        return

    def do_HEAD(self):
        self.__class__.requests.append(("HEAD", self.path, dict(self.headers)))
        status = self.scenario.get("head_status", 200)
        self.send_response(status)
        digest = self.scenario.get("digest")
        if digest:
            self.send_header("Docker-Content-Digest", digest)
        self.end_headers()

    def do_DELETE(self):
        self.__class__.requests.append(("DELETE", self.path, dict(self.headers)))
        self.send_response(self.scenario.get("delete_status", 202))
        self.end_headers()


@contextlib.contextmanager
def mock_registry(scenario):
    RegistryHandler.scenario = scenario
    RegistryHandler.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RegistryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", RegistryHandler.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_delete_image(registry_url=None, *, credentials=True, extra_env=None):
    env = os.environ.copy()
    for name in (
        "DOCKER_REGISTRY",
        "DOCKER_REPOSITORY",
        "TEMP_DOCKER_TAG",
        "DOCKER_USERNAME",
        "DOCKER_PASSWORD",
    ):
        env.pop(name, None)
    env.update(
        {
            "DOCKER_REPOSITORY": "ci/heavyedge/feature-models",
            "TEMP_DOCKER_TAG": "temp-tag",
        }
    )
    if registry_url is not None:
        env["DOCKER_REGISTRY"] = registry_url
    if credentials:
        env["DOCKER_USERNAME"] = "ci"
        env["DOCKER_PASSWORD"] = "ci-password"
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["sh", SCRIPT_PATH],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_contains(output, text):
    if text not in output:
        raise AssertionError(f"Expected to find {text!r} in output:\n{output}")


def assert_request_count(requests, method, expected):
    actual = sum(
        1 for request_method, _path, _headers in requests if request_method == method
    )
    if actual != expected:
        raise AssertionError(
            f"Expected {expected} {method} requests, got {actual}: {requests!r}"
        )


def test_missing_registry_skips():
    result = run_delete_image(registry_url=None)
    if result.returncode != 0:
        raise AssertionError(result)
    assert_contains(result.stderr, "environment is incomplete")


def test_missing_credentials_skips():
    with mock_registry({"head_status": 200}) as (registry_url, requests):
        result = run_delete_image(registry_url, credentials=False)
    if result.returncode != 0:
        raise AssertionError(result)
    assert_contains(result.stderr, "credentials are incomplete")
    if requests:
        raise AssertionError(f"Expected no registry requests, got {requests!r}")


def test_absent_tag_is_success():
    with mock_registry({"head_status": 404}) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode != 0:
        raise AssertionError(result)
    assert_contains(result.stdout, "already absent")
    assert_request_count(requests, "HEAD", 1)
    assert_request_count(requests, "DELETE", 0)


def test_head_error_fails_without_delete():
    with mock_registry({"head_status": 500}) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode == 0:
        raise AssertionError("Expected HEAD 500 to fail")
    assert_contains(result.stderr, "registry returned HTTP 500")
    assert_request_count(requests, "HEAD", 1)
    assert_request_count(requests, "DELETE", 0)


def test_missing_digest_fails_without_delete():
    with mock_registry({"head_status": 200}) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode == 0:
        raise AssertionError("Expected missing digest to fail")
    assert_contains(result.stderr, "did not return Docker-Content-Digest")
    assert_request_count(requests, "HEAD", 1)
    assert_request_count(requests, "DELETE", 0)


def test_delete_accepted_succeeds():
    digest = "sha256:abc123"
    scenario = {"head_status": 200, "digest": digest, "delete_status": 202}
    with mock_registry(scenario) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode != 0:
        raise AssertionError(result)
    assert_contains(result.stdout, "Deleted Docker image tag")
    assert_request_count(requests, "HEAD", 1)
    assert_request_count(requests, "DELETE", 1)
    delete_path = [path for method, path, _headers in requests if method == "DELETE"][0]
    if not delete_path.endswith(f"/manifests/{digest}"):
        raise AssertionError(f"Unexpected delete path: {delete_path}")


def test_delete_absent_succeeds():
    with mock_registry(
        {"head_status": 200, "digest": "sha256:alreadygone", "delete_status": 404}
    ) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode != 0:
        raise AssertionError(result)
    assert_contains(result.stdout, "Deleted Docker image tag")
    assert_request_count(requests, "DELETE", 1)


def test_delete_error_fails():
    with mock_registry(
        {"head_status": 200, "digest": "sha256:deletefail", "delete_status": 500}
    ) as (registry_url, requests):
        result = run_delete_image(registry_url)
    if result.returncode == 0:
        raise AssertionError("Expected DELETE 500 to fail")
    assert_contains(result.stderr, "registry returned HTTP 500")
    assert_request_count(requests, "DELETE", 1)


def main():
    tests = [
        test_missing_registry_skips,
        test_missing_credentials_skips,
        test_absent_tag_is_success,
        test_head_error_fails_without_delete,
        test_missing_digest_fails_without_delete,
        test_delete_accepted_succeeds,
        test_delete_absent_succeeds,
        test_delete_error_fails,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:
            failures += 1
            print(f"not ok - {test.__name__}: {error}", file=sys.stderr)
        else:
            print(f"ok - {test.__name__}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
