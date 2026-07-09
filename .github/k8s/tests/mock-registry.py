#!/usr/bin/env python3
import http.server


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def do_HEAD(self):
        print(f"HEAD {self.path}", flush=True)
        self.send_response(200)
        self.send_header("Docker-Content-Digest", "sha256:ci-cleanup")
        self.end_headers()

    def do_DELETE(self):
        print(f"DELETE {self.path}", flush=True)
        self.send_response(202)
        self.end_headers()


http.server.ThreadingHTTPServer(("0.0.0.0", 5000), Handler).serve_forever()
