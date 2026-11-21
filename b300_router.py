"""Loopback-only OpenAI chat router for the active four-B300 instance."""
import argparse
from contextlib import contextmanager
import fcntl
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import stat
import threading
import time


def active_slot(state, base_port):
    # Endpoints are derived from the configured port pair, never taken from a URL.
    with (state / "current.json").open("rb") as f:
        raw = f.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Invalid rotation state size")
    pointer = json.loads(raw)
    active = pointer["active_group"]
    if active not in ("a", "b"):
        raise ValueError("Invalid active group")
    for slot, port in (("a", base_port), ("b", base_port + 1)):
        recorded = pointer["groups"][slot]["port"]
        if type(recorded) is not int or recorded != port:
            raise ValueError("Rotation state disagrees with configured backend ports")
    return active


@contextmanager
def leased_port(state, base_port):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        slot = active_slot(state, base_port)
        fd = os.open(state / f"slot-{slot}.lease", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+b") as lease:
            if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
                raise ValueError("Invalid slot lease")
            try:
                fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            # A switch could have happened between the first read and flock().
            if active_slot(state, base_port) == slot:
                yield base_port if slot == "a" else base_port + 1
                return
    raise RuntimeError("Active slot is draining")


class RouterServer(ThreadingHTTPServer):
    # Let in-flight streams complete when serve_forever ends.
    daemon_threads = False
    request_queue_size = 128

    def __init__(self, port, state, base_port, *, max_body=32 << 20, max_requests=64,
                 timeout=300, max_duration=3600):
        if not 1 <= base_port < 65535 or port in (base_port, base_port + 1):
            raise ValueError("Router and the two backend ports must be distinct")
        if min(max_body, max_requests, timeout, max_duration) <= 0:
            raise ValueError("Router limits must be positive")
        self.state = Path(state)
        self.base_port = base_port
        self.max_body = max_body
        self.timeout_seconds = timeout
        self.max_duration = max_duration
        self.capacity = threading.BoundedSemaphore(max_requests)
        super().__init__(("127.0.0.1", port), RouterHandler)

    def process_request(self, request, client_address):
        if not self.capacity.acquire(blocking=False):
            try:
                request.settimeout(0.25)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.capacity.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.capacity.release()


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "glm-route"

    def setup(self):
        self.request.settimeout(min(self.server.timeout_seconds, 30))
        super().setup()

    def log_request(self, code="-", size="-"):
        # Deliberately exclude user-controlled query strings, credentials, and prompt bodies.
        self.log_message("%s %s", self.command, code)

    def failure(self, status, message):
        self.close_connection = True
        body = json.dumps({"error": {"type": "router_error", "message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.route("/v1/models")

    def do_POST(self):
        self.route("/v1/chat/completions")

    def route(self, allowed_path):
        self.close_connection = True
        if self.path != allowed_path:
            return self.failure(404, "Unknown API path")
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding"):
            return self.failure(400, "Send a Content-Length instead of transfer-encoding the request")
        if len(lengths) > 1 or (lengths and not lengths[0].isascii()) or (
                lengths and (not lengths[0].isdigit() or len(lengths[0]) > 19)):
            return self.failure(400, "Invalid Content-Length")
        length = int(lengths[0]) if lengths else 0
        if self.command == "POST" and not lengths:
            return self.failure(411, "Content-Length required")
        if length > self.server.max_body:
            return self.failure(413, "Request body exceeds configured limit")
        if self.command == "GET" and length:
            return self.failure(400, "Unexpected request body")
        deadline = time.monotonic() + self.server.max_duration

        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Request duration exceeded")
            return min(self.server.timeout_seconds, left)

        body = bytearray()
        try:
            while len(body) < length:
                self.connection.settimeout(remaining())
                chunk = self.rfile.read1(min(65536, length - len(body)))
                if not chunk:
                    return self.failure(400, "Incomplete HTTP body")
                body.extend(chunk)
        except TimeoutError:
            return self.failure(408, "Request body timed out")

        headers_sent = False
        waiting_for_slot = True
        try:
            with leased_port(self.server.state, self.server.base_port) as port:
                waiting_for_slot = False
                upstream = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining())
                try:
                    headers = {name: self.headers[name] for name in
                               ("Authorization", "Content-Type", "Accept", "X-Request-ID")
                               if name in self.headers}
                    upstream.connect()
                    transport = upstream.sock
                    upstream.request(self.command, allowed_path, body if self.command == "POST" else None,
                                     headers=headers)
                    with upstream.getresponse() as response:
                        self.send_response(response.status)
                        for name in ("Content-Type", "Content-Encoding", "Cache-Control", "Retry-After", "X-Request-ID"):
                            value = response.getheader(name)
                            if value and "\r" not in value and "\n" not in value:
                                self.send_header(name, value)
                        if response.length is not None:
                            self.send_header("Content-Length", str(response.length))
                        self.send_header("Connection", "close")
                        headers_sent = True
                        self.end_headers()
                        while True:
                            transport.settimeout(remaining())
                            chunk = response.read1(65536)
                            if not chunk:
                                break
                            self.connection.settimeout(remaining())
                            self.wfile.write(chunk)
                finally:
                    upstream.close()
        except (OSError, RuntimeError, KeyError, ValueError, TypeError, http.client.HTTPException) as error:
            # Replay could duplicate a completion. After headers the only safe failure is EOF.
            if not headers_sent:
                if waiting_for_slot:
                    return self.failure(503, "No routable active instance")
                if isinstance(error, TimeoutError):
                    return self.failure(504, "Active instance timed out")
                return self.failure(502, "Active instance unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-port", type=int, default=8100, help="Origin ports are N and N+1")
    parser.add_argument("--max-body-mib", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=300, help="Upstream idle timeout in seconds")
    parser.add_argument("--max-duration", type=int, default=3600, help="Total request budget in seconds")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        with RouterServer(args.port, args.state, args.base_port, max_body=args.max_body_mib << 20,
                          max_requests=args.max_requests, timeout=args.timeout,
                          max_duration=args.max_duration) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.exit(1, f"Router error: {error}\n")


if __name__ == "__main__":
    main()
