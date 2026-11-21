"""Check a merged text GGUF on one physical NVIDIA B300 with llama-server."""

import argparse
from contextlib import contextmanager
import csv
import ctypes as ct
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import stat
import threading
import time
import uuid

from b300_candidate import inspect_candidate


MIB = 1024**2
GENERATED = 32


def gpu_info(selector):
    query = "uuid,name,memory.total,memory.used,mig.mode.current,driver_version"
    result = subprocess.run(
        ["nvidia-smi", "--id=" + selector, "--query-gpu=" + query,
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError("Expected exactly one physical GPU")
    device, name, total, used, mig, driver = rows[0]
    if not re.search(r"\bB300\b", name) or mig.strip().lower() != "disabled":
        raise ValueError("The measurement requires a physical B300 with MIG disabled")
    uuid.UUID(device.removeprefix("GPU-"))
    return {"uuid": device, "name": name, "total_mib": int(total),
            "used_mib": int(used), "driver": driver}


@contextmanager
def reserve_device_memory(device_uuid, size):
    """Hold real HBM throughout loading and decoding, including unsampled peaks."""
    driver = ct.CDLL("libcuda.so.1")

    def call(name, *args):
        function = getattr(driver, name, None)
        if function is None:
            raise RuntimeError(f"Missing CUDA driver entry point {name}")
        rc = function(*args)
        if rc:
            raise RuntimeError(f"{name} failed (CUDA driver status {rc})")

    call("cuInit", 0)
    count = ct.c_int()
    call("cuDeviceGetCount", ct.byref(count))
    wanted = uuid.UUID(device_uuid.removeprefix("GPU-")).bytes
    for ordinal in range(count.value):
        device = ct.c_int()
        call("cuDeviceGet", ct.byref(device), ordinal)
        identity = (ct.c_ubyte * 16)()
        call("cuDeviceGetUuid_v2", ct.byref(identity), device)
        if bytes(identity) == wanted:
            break
    else:
        raise ValueError("Selected B300 is outside the process's visible CUDA devices")
    context = ct.c_void_p()
    pointer = ct.c_uint64()
    call("cuDevicePrimaryCtxRetain", ct.byref(context), device)
    try:
        call("cuCtxPushCurrent_v2", context)
        try:
            if size:
                call("cuMemAlloc_v2", ct.byref(pointer), ct.c_size_t(size))
                call("cuMemsetD8_v2", pointer, 0, ct.c_size_t(size))
                call("cuCtxSynchronize")
            yield
        finally:
            try:
                if pointer.value:
                    call("cuMemFree_v2", pointer)
            finally:
                call("cuCtxPopCurrent_v2", ct.byref(ct.c_void_p()))
    finally:
        call("cuDevicePrimaryCtxRelease_v2", device)


class LocalHTTP(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


def request(path, endpoint, payload, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Inference probe timed out")
    connection = LocalHTTP(path, remaining)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        connection.request("GET" if body is None else "POST", endpoint, body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status == 503 and endpoint == "/health":
            return {}
        if response.status != 200:
            raise RuntimeError(f"{endpoint} returned HTTP {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


def require_placement(log):
    layers = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    if not layers or any(int(placed) == 0 or placed != total for placed, total in layers):
        raise RuntimeError("Missing evidence of complete model layer offload")
    buffers = re.findall(r"(\S+)\s+(model|KV|RS|rotation) buffer size\s*=\s*([\d.]+) MiB", log)
    found = set()
    for device, kind, amount in buffers:
        if float(amount) > 0:
            if device != "CUDA0":
                raise RuntimeError(f"{kind} buffer placed on {device}")
            found.add(kind)
    if "model" not in found or not found.intersection({"KV", "RS"}):
        raise RuntimeError("Runtime did not confirm CUDA0 weight and state buffers")
    return {"layers_on_gpu": int(layers[-1][0]), "device_buffers": sorted(found)}


def complete(path, context, deadline):
    tokens = request(path, "/tokenize", {"content": " A short example with numbers 21 and 42.",
                                          "add_special": False}, deadline)["tokens"]
    if not tokens or not all(type(t) is int and t >= 0 for t in tokens):
        raise RuntimeError("Could not construct an exact-length token prompt")
    prompt_length = context - GENERATED
    prompt = (tokens * ((prompt_length + len(tokens) - 1) // len(tokens)))[:prompt_length]
    result = request(path, "/completion", {
        "prompt": prompt, "n_predict": GENERATED, "ignore_eos": True,
        "cache_prompt": False, "temperature": 0, "id_slot": 0,
    }, deadline)
    if (result.get("tokens_evaluated") != prompt_length or
            result.get("tokens_predicted") != GENERATED or result.get("stop_type") != "limit"):
        raise RuntimeError("Server did not complete the requested prefill and decode")
    return {"prefill_tokens": prompt_length, "generated_tokens": GENERATED}


def exec_server(expected_parent, argv):
    # A killed launcher must not leave a server running after the HBM guard disappears.
    if ct.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) or os.getppid() != expected_parent:
        raise SystemExit("Could not bind llama-server lifetime to the reservation holder")
    os.execv(argv[0], argv)


@contextmanager
def probe(binary, candidate, gpu, context, timeout):
    with tempfile.TemporaryDirectory(prefix="glm-fit-") as directory:
        path = Path(directory) / "llama.sock"
        log_path = Path(directory) / "server.log"
        profile = ["--alias", "glm-5.3-flash", "--parallel", "1", "--ctx-size", str(context), "--device", "CUDA0",
                   "--split-mode", "none", "--gpu-layers", "all", "--n-cpu-moe", "0",
                   "--override-tensor", ".*=CUDA0", "--no-host", "--fit", "off",
                   "--load-mode", "none", "--lazy-mode", "off", "--no-context-shift",
                   "--cache-ram", "0", "--warmup"]
        command = [str(binary), "--model", str(candidate.entrypoint), "--host", str(path), *profile]
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("LLAMA_", "GGML_"))}
        env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_MANAGED_FORCE_DEVICE_ALLOC="1")
        deadline = time.monotonic() + timeout
        stop = threading.Event()
        peaks = []
        sample_errors = []

        def monitor():
            try:
                while not stop.is_set():
                    peaks.append(gpu_info(gpu["uuid"])["used_mib"])
                    stop.wait(0.5)
            except Exception as error:
                sample_errors.append(str(error))

        sampler = threading.Thread(target=monitor)
        with log_path.open("wb") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_exec-server",
                                      str(os.getpid()), *command], env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            try:
                sampler.start()
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        tail = log_path.read_bytes()[-1200:].decode(errors="replace")
                        raise RuntimeError(f"llama-server exited during loading: {tail}")
                    try:
                        if request(path, "/health", None, deadline).get("status") == "ok":
                            break
                    except (FileNotFoundError, ConnectionRefusedError):
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError("llama-server startup timed out")
                work = complete(path, context, deadline)
                peaks.append(gpu_info(gpu["uuid"])["used_mib"])
                if child.poll() is not None:
                    raise RuntimeError("llama-server stopped during inference")
                placement = require_placement(log_path.read_text(errors="replace"))
                stop.set()
                sampler.join()
                if sample_errors or not peaks:
                    raise RuntimeError(f"Incomplete GPU telemetry: {sample_errors}")
                yield {**work, **placement, "sampled_device_peak_mib_including_reservation": max(peaks),
                       "samples": len(peaks), "context_tokens": context, "parallel_slots": 1,
                       "server_flags": profile}, child, path
            finally:
                stop.set()
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                sampler.join()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MIB), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def measure(args):
    if args.context <= GENERATED or args.reserve_mib < 0 or args.timeout <= 0:
        raise ValueError("Expect context > 32, reserve >= 0 MiB, timeout > 0 s")
    gpu = gpu_info(args.gpu)
    capacity = gpu["total_mib"] * MIB
    reserve = args.reserve_mib * MIB
    candidate = inspect_candidate(args.model, capacity_bytes=capacity, reserve_bytes=reserve)
    manifest = candidate.to_dict()
    binary_path = shutil.which(args.llama_server)
    if not binary_path:
        raise FileNotFoundError(f"Cannot find {args.llama_server}")
    binary = Path(binary_path).resolve(strict=True)
    binary_hash = digest(binary)
    # If the driver reports more bytes than the nominal cap, withhold those too.
    physical_reservation = capacity - manifest["capacity_bytes"] + reserve
    with reserve_device_memory(gpu["uuid"], physical_reservation), probe(
            binary, candidate, gpu, args.context, args.timeout) as (result, child, path):
        if digest(binary) != binary_hash or inspect_candidate(
                args.model, capacity_bytes=capacity, reserve_bytes=reserve).sha256 != candidate.sha256:
            raise ValueError("Candidate or server binary changed during measurement")
        if child.poll() is not None:
            raise RuntimeError("Server exited during final verification")
        yield {"status": "passed", "candidate": manifest, "gpu": gpu,
               "model_sha256": candidate.sha256, "server_sha256": binary_hash,
               "server_path": str(binary), "physical_reservation_bytes": physical_reservation,
               **result}, child, path


def publish(path, report):
    # Atomic no-replace publication keeps concurrent measurements independent.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent) as staging:
        json.dump(report, staging, indent=2, ensure_ascii=False)
        staging.write("\n")
        staging.flush()
        os.fsync(staging.fileno())
        os.link(staging.name, path)


@contextmanager
def expose(socket_link, private_socket):
    if socket_link is None:
        yield
        return
    socket_link.symlink_to(private_socket)
    try:
        yield
    finally:
        if socket_link.is_symlink() and os.readlink(socket_link) == str(private_socket):
            socket_link.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or full GPU UUID")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--reserve-mib", type=int, default=8192)
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds for server load and decode")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--serve", action="store_true", help="Keep the measured server and VRAM reservation alive")
    parser.add_argument("--socket", type=Path, help="New .sock path in a private directory for --serve")
    args = parser.parse_args(argv)
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("--report needs a new filename under an existing directory")
    if args.serve != bool(args.socket):
        parser.error("--serve and --socket must be used together")
    if args.socket:
        args.socket = args.socket.absolute()
        if (not args.socket.parent.is_dir() or args.socket.suffix != ".sock" or
                os.path.lexists(args.socket) or len(os.fsencode(args.socket)) >= 108):
            parser.error("--socket must be a new .sock path of at most 107 bytes")
        permissions = args.socket.parent.stat()
        if permissions.st_uid != os.geteuid() or stat.S_IMODE(permissions.st_mode) != 0o700:
            parser.error("--socket requires a directory owned by this user with mode 0700")
    success_published = False
    try:
        with measure(args) as (report, child, path), expose(args.socket, path):
            publish(args.report, report)
            success_published = True
            print(args.report, flush=True)
            if args.serve:
                print(f"llama-server Unix socket: {args.socket}", flush=True)
                raise RuntimeError(f"Serving process exited with status {child.wait()}")
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, http.client.HTTPException,
            subprocess.SubprocessError) as error:
        if not success_published:
            try:
                publish(args.report, {"status": "failed", "reason": str(error)})
            except OSError as report_error:
                print(f"Could not publish fit report: {report_error}", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["_exec-server"]:
        exec_server(int(sys.argv[2]), sys.argv[3:])
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
