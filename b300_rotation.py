"""Alternate two four-B300 groups between live BF16 inference and offline improvement."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import b300_fleet as fleet
from b300_lineage import WeightSnapshot, stable_digest
from b300_train import available_devices, interrupt_on_termination, reserve_devices, validate_model


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def publish(path, contents):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as target:
        json.dump(contents, target, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    tmp.replace(path)
    with_dir = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(with_dir)
    finally:
        os.close(with_dir)


def model_id(weights):
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Missing weight digests")
    encoded = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def controller(state):
    with (state / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock.fileno()


def invoke(module, args, lock, devices=None):
    env = dict(os.environ)
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    # A child that outlives a hard controller crash retains its lock. Recovery can
    # only act after that job finishes. Normal termination forwards to its process group.
    with subprocess.Popen([sys.executable, "-m", module, *map(str, args)], env=env,
                          pass_fds=(lock,), start_new_session=True) as child:
        try:
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, child.args)


def start_service(slot, model, lock):
    invoke("b300_fleet", ["start", "--precision", "bf16", "--replicas", 1,
                          "--name", slot["name"], "--port", slot["port"], "--model", model],
           lock, slot["devices"])


def verify_service(slot, model):
    ids = fleet.containers(slot["name"])
    if len(ids) != 1:
        raise ValueError(f"Expected one managed NIM container for {slot['name']}")
    info = json.loads(subprocess.check_output(["docker", "inspect", ids[0]], text=True))[0]
    mounts = [m for m in info["Mounts"] if m["Destination"] == "/weights/snapshots/base"]
    ports = info["NetworkSettings"]["Ports"].get("8000/tcp") or []
    device_ids = [i for req in info["HostConfig"]["DeviceRequests"] for i in (req["DeviceIDs"] or [])]
    if (not info["State"]["Running"] or len(mounts) != 1 or mounts[0]["RW"]
            or Path(mounts[0]["Source"]).resolve() != Path(model).resolve()
            or ports != [{"HostIp": "127.0.0.1", "HostPort": str(slot["port"])}]
            or set(device_ids) != set(slot["devices"])
            or "NIM_MODEL_PROFILE=" + fleet.PROFILES["bf16"][1] not in info["Config"]["Env"]
            or not fleet.ready(slot["port"])):
        raise ValueError("Live NIM ownership, BF16 profile, checkpoint, GPU set, port or readiness mismatch")


def stop_slot(state, slot_id, slot, lock):
    # The router holds a shared lease from route selection through the last byte
    # of each response. Never remove or replace the lease inode during rotation.
    with (state / f"slot-{slot_id}.lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        invoke("b300_fleet", ["stop", "--name", slot["name"]], lock)


def recover_pending(state, lock):
    pending = state / "pending.json"
    if not pending.exists():
        return
    journal = read(pending)
    pointer = read(state / "current.json") if (state / "current.json").exists() else {}
    if pointer.get("publication") == journal["attempt"]:
        old = journal["old"]
        if old is not None:
            stop_slot(state, old, pointer["groups"][old], lock)
    else:
        if pointer.get("publication") != journal["parent"]:
            raise ValueError("Unrelated active generation; refusing to retire a slot")
        target = journal["target"]
        if pointer.get("active_group") == target:
            raise ValueError("Recovered target is already the active slot")
        stop_slot(state, target, journal["groups"][target], lock)
    pending.unlink()


def endpoint(slot):
    return f"http://127.0.0.1:{slot['port']}/v1"


def initialize(args, state, lock):
    if (state / "current.json").exists():
        raise ValueError("Rotation is already initialized")
    if not 1 <= args.port <= 65534:
        raise ValueError("Both slot ports must fit within 1..65535")
    base = args.model.resolve(strict=True)
    validate_model(base)
    devices = available_devices(8)[:8]
    namespace = uuid.uuid4().hex[:16]
    groups = {slot: {"name": f"rot-{namespace}-{slot}", "devices": devices[n*4:n*4+4],
                     "port": args.port + n} for n, slot in enumerate(("a", "b"))}
    weights = WeightSnapshot(base)
    digest = model_id(weights.weight_hashes(read(base / "model.safetensors.index.json")))
    current = {"publication": namespace, "active_group": "a", "peer_group": "b", "groups": groups,
               "generation": 0, "active_endpoint": endpoint(groups["a"]), "active_model": str(base),
               "model_sha256": digest, "config_sha256": stable_digest(base / "config.json"),
               "export": None, "fit": None}
    publish(state / "pending.json", {"attempt": namespace, "parent": None, "old": None,
                                     "target": "a", "groups": groups})
    try:
        start_service(groups["a"], base, lock)
        verify_service(groups["a"], base)
        weights.check()
        publish(state / "current.json", current)
    finally:
        recover_pending(state, lock)


def export_and_measure(model, workspace, llama_dir, slot, lock):
    export = workspace / "gguf"
    report = workspace / "fit.json"
    invoke("b300_export", ["--model", model, "--llama-dir", llama_dir, "--out", export], lock)
    gpu = slot["devices"][0]
    # No other cooperating launcher may claim this GPU while the fit starts.
    with reserve_devices(1, allowed={gpu}):
        invoke("b300_fit", ["--model", export / "candidate.gguf", "--gpu", gpu,
                            "--llama-server", llama_dir / "build/bin/llama-server", "--report", report],
               lock, [gpu])
    manifest = read(export / "export-manifest.json")
    result = read(report)
    if (manifest["source_model"] != str(model) or result.get("status") != "passed"
            or result.get("gpu", {}).get("uuid") != gpu
            or not result.get("model_sha256") or result["model_sha256"] != manifest["candidate"]["sha256"]):
        raise ValueError("Single-B300 evidence does not match the candidate")
    return export, report, model_id(manifest["source_weight_sha256"])


def run_round(args, state, lock):
    current = read(state / "current.json")
    old, new = current["active_group"], current["peer_group"]
    groups = current["groups"]
    slot = groups[new]
    base = Path(current["active_model"])
    checks = args.checks.resolve(strict=True)
    data = args.data.resolve(strict=True)
    llama = args.llama_dir.resolve(strict=True)
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.extra_width < 0 or args.extra_width % 256:
        raise ValueError("--extra-width must be a nonnegative multiple of 256")
    verify_service(groups[old], base)
    if fleet.containers(slot["name"]):
        raise ValueError("Standby slot still owns a container; recover previous rotation first")
    if stable_digest(base / "config.json") != current["config_sha256"]:
        raise ValueError("Active model configuration changed")
    original = WeightSnapshot(base)
    checks_digest = stable_digest(checks)
    attempt = uuid.uuid4().hex
    work = state / "rounds" / attempt
    work.mkdir(parents=True)
    publish(state / "pending.json", {"attempt": attempt, "parent": current["publication"], "old": old,
                                     "target": new, "groups": groups})
    try:
        source = base
        if args.extra_width:
            if current["export"] and current["fit"]:
                base_export, base_fit = Path(current["export"]), Path(current["fit"])
                parent_digest = model_id(read(base_export / "export-manifest.json")["source_weight_sha256"])
            else:
                baseline = work / "baseline"
                baseline.mkdir()
                base_export, base_fit, parent_digest = export_and_measure(base, baseline, llama, slot, lock)
            if parent_digest != current["model_sha256"]:
                raise ValueError("Growth parent does not match the live generation")
            source = work / "grown"
            invoke("b300_checkpoint", ["grow", "--model", base, "--export", base_export, "--fit", base_fit,
                                       "--out", source, "--extra-width", args.extra_width], lock)
        train = work / "train"
        options = ["--steps", args.steps] if args.steps else []
        invoke("b300_train", ["--model", source, "--data", data, "--out", train,
                              "--gpus", 4, *options], lock, slot["devices"])
        run = read(train / "run.json")
        if run["status"] != "trained" or set(run["devices"]) != set(slot["devices"]):
            raise ValueError("The training receipt does not match the standby group")
        if source == base and model_id(run["source_weight_sha256"]) != current["model_sha256"]:
            raise ValueError("Training base differs from the live generation")
        candidate = work / "merged"
        invoke("b300_checkpoint", ["merge", "--model", source, "--run", train, "--out", candidate], lock)
        snapshot = WeightSnapshot(candidate)
        export, fit, digest = export_and_measure(candidate, work, llama, slot, lock)
        start_service(slot, candidate, lock)
        verify_service(groups[old], base)
        verify_service(slot, candidate)
        report = work / "evaluation.json"
        invoke("b300_evaluate", ["--base-port", groups[old]["port"], "--candidate-port", slot["port"],
                                 "--base-sha256", current["model_sha256"], "--candidate-sha256", digest,
                                 "--training-data", train / "inputs/train.jsonl", "--checks", checks,
                                 "--report", report], lock)
        result = read(report)
        expected = {"status": "passed", "base_port": groups[old]["port"], "candidate_port": slot["port"],
                    "base_sha256": current["model_sha256"], "candidate_sha256": digest, "regressions": 0,
                    "training_data_sha256": run["training_data_sha256"], "checks_sha256": checks_digest}
        if any(result.get(k) != v for k, v in expected.items()) or result["target_fixes"] < 1 or result["controls_passed"] < 1:
            raise ValueError("Quality evaluation did not establish an improvement for these snapshots")
        original.check()
        snapshot.check()
        verify_service(slot, candidate)
        publish(state / "current.json", {
            **current, "publication": attempt, "generation": current["generation"] + 1,
            "active_group": new, "peer_group": old, "active_endpoint": endpoint(slot),
            "active_model": str(candidate), "model_sha256": digest,
            "config_sha256": stable_digest(candidate / "config.json"),
            "evaluation": str(report), "export": str(export), "fit": str(fit),
        })
    finally:
        recover_pending(state, lock)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for name in ("init", "round", "recover", "status"):
        action = actions.add_parser(name)
        action.add_argument("--state", type=Path, required=True)
        if name == "init":
            action.add_argument("--model", type=Path, required=True)
            action.add_argument("--port", type=int, default=8100)
        if name == "round":
            for path in ("data", "checks", "llama-dir"):
                action.add_argument("--" + path, type=Path, required=True)
            action.add_argument("--steps", type=int)
            action.add_argument("--extra-width", type=int, default=0)
    args = cli.parse_args()
    state = args.state.resolve()
    try:
        if args.action == "status":
            print(json.dumps({"current": read(state / "current.json") if (state / "current.json").exists() else None,
                              "pending": read(state / "pending.json") if (state / "pending.json").exists() else None},
                             indent=2))
            return
        if args.action == "init":
            state.mkdir(parents=True, mode=0o700, exist_ok=True)
        with controller(state) as lock:
            recover_pending(state, lock)
            if args.action == "init":
                initialize(args, state, lock)
            elif args.action == "round":
                run_round(args, state, lock)
            elif (state / "current.json").exists():
                current = read(state / "current.json")
                slot = current["groups"][current["active_group"]]
                if not fleet.containers(slot["name"]):
                    start_service(slot, current["active_model"], lock)
                verify_service(slot, current["active_model"])
        if (state / "current.json").exists():
            print(json.dumps(read(state / "current.json"), indent=2))
    except KeyboardInterrupt:
        cli.exit(130, "Rotation interrupted.\n")
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        cli.exit(1, f"Rotation failed: {exc}\n")


if __name__ == "__main__":
    interrupt_on_termination()
    main()
