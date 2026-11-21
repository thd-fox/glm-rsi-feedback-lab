"""Launch an isolated single-node GLM-5.3-Flash QLoRA job from approved SFT JSONL."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import shlex
import signal
import stat
import subprocess
from contextlib import ExitStack, contextmanager
from importlib.resources import files
from pathlib import Path
from typing import BinaryIO

from b300_lineage import WeightSnapshot


RECIPE = files("glm_feedback_recipes").joinpath("b300-train.yml")
DEFAULT_IMAGE = "axolotlai/axolotl:main-latest"


def validate_snapshot_files(model: Path, index: bytes) -> Path | None:
    shards = set(json.loads(index)["weight_map"].values())
    if not shards or any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise ValueError("Expected local safetensors shard filenames in the weight index")
    for name in ("tokenizer.json", "tokenizer_config.json", *shards):
        if not (model / name).is_file():
            raise ValueError(f"Missing model file: {name}")

    # Preserve the relative symlinks used by huggingface_hub's snapshot layout.
    cache_blobs = model.parent.parent / "blobs"
    blobs = cache_blobs if model.parent.name == "snapshots" and cache_blobs.is_dir() else None
    for file in model.iterdir():
        if file.is_symlink() and not file.resolve().is_relative_to(model):
            if blobs is None or not os.readlink(file).startswith("../../blobs/") or not file.resolve().is_relative_to(blobs):
                raise ValueError(f"External model symlink needs a materialized local checkpoint: {file.name}")
    return blobs


def validate_model(model: Path) -> tuple[dict, Path | None]:
    config = (model / "config.json").read_bytes()
    metadata = json.loads(config)
    text_config = metadata.get("text_config", {})
    if metadata.get("model_type") != "glm5_next" or text_config.get("dtype", text_config.get("torch_dtype")) != "bfloat16":
        raise ValueError("Expected a BF16 GLM-5.3-Flash (glm5_next) checkpoint")
    if metadata.get("quantization_config") or text_config.get("quantization_config"):
        raise ValueError("FSDP QLoRA requires the BF16 source checkpoint")
    index = (model / "model.safetensors.index.json").read_bytes()
    blobs = validate_snapshot_files(model, index)
    return {
        "base_model": str(model),
        "hf_snapshot": model.name if blobs is not None else None,
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "weight_index_sha256": hashlib.sha256(index).hexdigest(),
    }, blobs


def snapshot_dataset(source: Path, output: BinaryIO | None = None) -> dict:
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                messages = record["messages"]
                if not isinstance(messages, list) or len(messages) != 2:
                    raise ValueError
                for message, role in zip(messages, ("user", "assistant")):
                    if message["role"] != role or not isinstance(message["content"], str) or not message["content"].strip():
                        raise ValueError
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"Line {line_no}: expected a single-turn SFT record with a user and assistant text") from exc
            # Export only the two verified message fields. Audit/feedback fields
            # accidentally present alongside them never enter the container.
            clean = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
            encoded = (json.dumps(clean, ensure_ascii=False) + "\n").encode("utf-8")
            if output is not None:
                output.write(encoded)
            digest.update(encoded)
            count += 1
    if not count:
        raise ValueError("Empty SFT dataset")
    return {"rows": count, "training_data_sha256": digest.hexdigest()}


def available_devices(count: int) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    busy = {line.strip() for line in active.stdout.splitlines()}
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = {d.strip() for d in assigned.split(",")} if assigned is not None else None
    devices = []
    for index, uuid, name in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if visible is not None and index not in visible and uuid not in visible:
            continue
        if uuid not in busy and re.search(r"\bB300\b", name.replace("_", " "), re.IGNORECASE):
            devices.append(uuid)
    if len(devices) < count:
        raise ValueError(f"Requested {count} B300 GPUs. Found {len(devices)} assigned cards without an active compute process.")
    return devices


@contextmanager
def reserve_devices(count: int, *, directory: Path = Path("/tmp"), allowed=None):
    """Hold host-wide per-UUID locks, skipping devices another launcher just took."""
    selected = []
    with ExitStack() as held:
        for device in available_devices(count):
            if allowed is not None and device not in allowed:
                continue
            if not re.fullmatch(r"GPU-[a-f0-9-]+", device, re.IGNORECASE):
                raise ValueError(f"Invalid nvidia-smi GPU UUID: {device}")
            path = directory / f"glm-feedback-lab-{device}.lock"
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            try:
                fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                fd = os.open(path, flags)
            else:
                os.fchmod(fd, 0o644)  # Separate host users can lock an existing file read-only.
            held.enter_context(os.fdopen(fd, "rb"))
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"Expected a regular GPU lock file at {path}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            selected.append(device)
            if len(selected) == count:
                # A fleet may have finished startup between enumeration and locking.
                if not set(selected).issubset(available_devices(count)):
                    raise ValueError("GPU occupancy changed during allocation; retry the run")
                yield selected
                return
        raise ValueError(f"Only {len(selected)} of {count} requested B300 GPUs could be reserved")


def interrupt_on_termination():
    """Let context managers clean up detached Docker resources on normal host termination."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)


def bind(source: Path, target: str, *, readonly: bool = True) -> list[str]:
    spec = ["type=bind", f"src={source}", f"dst={target}"]
    if readonly:
        spec.append("readonly")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(spec)
    return ["--mount", buffer.getvalue().rstrip("\n")]


def launch_command(model: Path, blobs: Path | None, run: Path, devices: list[str], image: str) -> list[str]:
    cmd = [
        "docker", "run", "--rm", "--pull=never", "--init", "--cidfile", str(run / "container.cid"), "--network=none",
        "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,mode=1777",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--shm-size=16g",
        "--ulimit", "memlock=-1:-1", "--user", f"{os.getuid()}:{os.getgid()}",
        "--gpus", '"device=' + ",".join(devices) + '"',
        "--workdir", "/out", "-e", "HOME=/out", "-e", "HF_HOME=/out/cache/hf",
        "-e", "HF_HUB_OFFLINE=1", "-e", "HF_DATASETS_OFFLINE=1", "-e", "WANDB_MODE=disabled",
        "-e", "GLOO_SOCKET_IFNAME=lo", "-e", "NCCL_SOCKET_IFNAME=lo",
    ]
    cmd += bind(model, "/weights/snapshots/base")
    if blobs is not None:
        cmd += bind(blobs, "/weights/blobs")
    cmd += bind(run / "inputs", "/inputs")
    cmd += bind(run / "artifacts", "/out", readonly=False)
    return cmd + ["--entrypoint", "torchrun", image, "--standalone", "--nnodes=1",
                  f"--nproc-per-node={len(devices)}", "-m", "axolotl.cli.train", "/inputs/train.yml"]


def write_manifest(run: Path, manifest: dict) -> None:
    tmp = run / "run.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(run / "run.json")


def train(args: argparse.Namespace) -> None:
    model = args.model.resolve(strict=True)
    data = args.data.resolve(strict=True)
    run = args.out.resolve()
    if run.exists() or run.is_relative_to(model):
        raise ValueError("--out must be a new run directory outside the base checkpoint")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    weights = WeightSnapshot(model)
    provenance, blobs = validate_model(model)
    recipe = RECIPE.read_text(encoding="utf-8")
    if args.steps:
        recipe += f"\nmax_steps: {args.steps}\n"
    if args.dry_run:
        print(json.dumps(snapshot_dataset(data), indent=2))
        print(shlex.join(launch_command(model, blobs, run, list(map(str, range(args.gpus))), args.image)))
        return

    # Pin the resolved local image ID even if the caller supplied a moving tag.
    image = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError(f"Could not resolve the local container image: {args.image}")
    run.mkdir(mode=0o700, parents=True, exist_ok=False)
    inputs = run / "inputs"
    inputs.mkdir()
    artifacts = run / "artifacts"
    artifacts.mkdir()
    manifest = {**provenance, "image_id": image, "source_data": str(data),
                "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(), "status": "preparing"}
    write_manifest(run, manifest)
    try:
        (inputs / "train.yml").write_text(recipe, encoding="utf-8")
        with (inputs / "train.jsonl").open("xb") as out:
            manifest.update(snapshot_dataset(data, out))
        index = json.loads((model / "model.safetensors.index.json").read_bytes())
        manifest["source_weight_sha256"] = weights.weight_hashes(index)
        with reserve_devices(args.gpus) as devices:
            manifest.update(status="training", devices=devices)
            write_manifest(run, manifest)
            try:
                subprocess.run(launch_command(model, blobs, run, devices, image), check=True)
            except (Exception, KeyboardInterrupt):
                cidfile = run / "container.cid"
                if cidfile.exists():
                    cid = cidfile.read_text().strip()
                    if re.fullmatch(r"[a-f0-9]{64}", cid):
                        subprocess.run(["docker", "rm", "-f", cid], check=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raise
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (artifacts / "adapter" / name).is_file():
                raise ValueError(f"Training exited without the expected PEFT adapter file: {name}")
        weights.check()
    except (Exception, KeyboardInterrupt):
        manifest["status"] = "failed"
        write_manifest(run, manifest)
        raise
    manifest["status"] = "trained"
    write_manifest(run, manifest)
    print(artifacts / "adapter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local GLM-5.3-Flash BF16 snapshot")
    parser.add_argument("--data", type=Path, required=True, help="approved SFT JSONL from glm_feedback")
    parser.add_argument("--out", type=Path, required=True, help="new private run directory")
    parser.add_argument("--gpus", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--steps", type=int, help="optimizer steps (default: one epoch)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="locally available Axolotl Docker image")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print command without creating a job")
    try:
        train(parser.parse_args())
    except KeyboardInterrupt:
        parser.exit(130, "Training interrupted.\n")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Training setup or run failed: {exc}\n")


if __name__ == "__main__":
    interrupt_on_termination()
    main()
