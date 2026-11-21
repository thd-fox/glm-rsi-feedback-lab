"""Plan, start and stop local GLM-5.3-Flash NIM replicas on an eight-B300 host."""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from b300_train import bind, interrupt_on_termination, reserve_devices, validate_snapshot_files


IMAGE = "nvcr.io/nim/nvidia/sglang-model-free-nim:2.1.4-variant-glm-5.3-flash"
# NVIDIA NIM 2.1.4 support matrix, model class Glm5NextForConditionalGeneration.
PROFILES = {
    "nvfp4": (2, "caeafff184c5ddc019a4dec380841200e5ac8eceebd39fcbd66c58258b794dfb"),
    "fp8": (4, "6780254ec7375998309aa95c0fd5aa5ce37b66b5834cabfa5f09e40a3713b876"),
    "bf16": (4, "4bb955cd9a620abfef9439ed2113a386d2a1a9c7c316d77b12807f2b3b9be349"),
}
FLEET_LABEL = "org.glm-feedback-lab.nim.fleet"
OWNER_LABEL = "org.glm-feedback-lab.nim.owner"
ALIAS = "glm-5.3-flash"


def plan(precision: str, replicas: int | None, port: int) -> dict:
    width, profile = PROFILES[precision]
    count = 8 // width if replicas is None else replicas
    if not 1 <= count <= 8 // width:
        raise ValueError(f"{precision} needs {width} B300s per replica; an 8-GPU server fits {8 // width}")
    if not (1 <= port and port + count - 1 <= 65535):
        raise ValueError("Port range must fit within 1..65535")
    return {
        "image": IMAGE, "precision": precision, "tensor_parallel": width,
        "profile": profile, "gpu_count": count * width, "model_alias": ALIAS,
        "replicas": [{"devices": list(range(i * width, (i + 1) * width)), "port": port + i,
                      "endpoint": f"http://127.0.0.1:{port + i}/v1"} for i in range(count)],
    }


def validate_checkpoint(model: Path, precision: str) -> Path | None:
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "glm5_next" or "Glm5NextForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Expected a GLM-5.3-Flash checkpoint with glm5_next architecture")
    qc = config.get("quantization_config") or {}
    text = config.get("text_config", {})
    formats = {
        "nvfp4": qc.get("quant_method") == "modelopt" and qc.get("quant_algo") == "NVFP4",
        "fp8": qc.get("quant_method") == "fp8",
        "bf16": not qc and text.get("dtype", text.get("torch_dtype")) == "bfloat16",
    }
    if not formats[precision]:
        raise ValueError(f"Local weights do not match the {precision} profile")
    return validate_snapshot_files(model, (model / "model.safetensors.index.json").read_bytes())


@contextmanager
def fleet_lock(name: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,48}", name):
        raise ValueError("Invalid fleet name (use lowercase letters, digits, dash, underscore or dot)")
    root = Path(tempfile.gettempdir()) / f"glm-feedback-fleets-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("Invalid permissions on the local fleet lock directory")
    with (root / name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def containers(name: str) -> list[str]:
    return subprocess.check_output([
        "docker", "ps", "-aq", "--no-trunc", "--filter", f"label={FLEET_LABEL}={name}",
        "--filter", f"label={OWNER_LABEL}={os.getuid()}",
    ], text=True).split()


def create_replica(name, index, devices, port, profile, image_id, model, blobs) -> str:
    cmd = [
        "docker", "create", "--pull=never", "--name", f"glmf-{name}-{index + 1}",
        "--label", f"{FLEET_LABEL}={name}", "--label", f"{OWNER_LABEL}={os.getuid()}",
        "--gpus", '"device=' + ",".join(devices) + '"', "--shm-size=64g",
        "--publish", f"127.0.0.1:{port}:8000", "--volume", "/opt/nim/.cache",
        "--env", "HF_HUB_OFFLINE=1", "--env", "NIM_MODEL_PATH=/weights/snapshots/base",
        "--env", f"NIM_MODEL_PROFILE={profile}", "--env", f"NIM_SERVED_MODEL_NAME={ALIAS}",
    ]
    cmd += bind(model, "/weights/snapshots/base")
    if blobs is not None:
        cmd += bind(blobs, "/weights/blobs")
    return subprocess.check_output(cmd + [image_id], text=True).strip()


def ready(port: int) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/v1/health/ready")
        return connection.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def wait_ready(ids: list[str], ports: list[int], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.Status}}", *ids], text=True
        ).split()
        if len(statuses) != len(ids) or any(state != "running" for state in statuses):
            raise RuntimeError(f"NIM replica exited before readiness. Container states: {statuses}")
        if all([ready(port) for port in ports]):
            return
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise RuntimeError("Timed out waiting for all NIM /v1/health/ready endpoints")


def remove(ids: list[str]) -> None:
    if ids:
        subprocess.run(["docker", "rm", "-fv", *ids], check=True, stdout=subprocess.DEVNULL)


def start(args, layout: dict) -> None:
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    model = args.model.resolve(strict=True)
    blobs = validate_checkpoint(model, args.precision)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Could not resolve the local NIM image")
    with fleet_lock(args.name), reserve_devices(layout["gpu_count"]) as devices:
        if containers(args.name):
            raise ValueError("This fleet name already exists; use stop or choose another name")
        ids = []
        try:
            for index, replica in enumerate(layout["replicas"]):
                replica["devices"] = [devices[i] for i in replica["devices"]]
                ids.append(create_replica(args.name, index, replica["devices"], replica["port"],
                                          layout["profile"], image_id, model, blobs))
            for cid in ids:
                subprocess.run(["docker", "start", cid], check=True, stdout=subprocess.DEVNULL)
            wait_ready(ids, [r["port"] for r in layout["replicas"]], args.timeout)
        except (Exception, KeyboardInterrupt):
            # docker create may have registered a container just before interruption
            # delivered the ID to this process. Include matching fleet-owned labels.
            remove(list(dict.fromkeys(ids + containers(args.name))))
            raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for action in ("plan", "start", "stop"):
        parser = actions.add_parser(action)
        parser.add_argument("--name", default="glm-fleet")
        if action in ("plan", "start"):
            parser.add_argument("--precision", choices=PROFILES, default="bf16")
            parser.add_argument("--replicas", type=int, help="default: fill all eight GPUs")
            parser.add_argument("--port", type=int, default=8100, help="first loopback API port")
        if action == "start":
            parser.add_argument("--model", type=Path, required=True, help="local complete checkpoint")
            parser.add_argument("--timeout", type=int, default=1800, help="readiness deadline in seconds")
    args = cli.parse_args()
    try:
        if args.action == "stop":
            with fleet_lock(args.name):
                remove(containers(args.name))
            return
        layout = plan(args.precision, args.replicas, args.port)
        if args.action == "start":
            start(args, layout)
        print(json.dumps(layout, indent=2))
    except KeyboardInterrupt:
        cli.exit(130, "Fleet operation interrupted.\n")
    except (KeyError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"Fleet error: {exc}\n")


if __name__ == "__main__":
    interrupt_on_termination()
    main()
