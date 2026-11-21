"""Export a merged local Hugging Face checkpoint to a single-card GGUF candidate."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from b300_candidate import B300_NOMINAL_BYTES, DEFAULT_RESERVE_BYTES, inspect_candidate


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_weights(source):
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text())["weight_map"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Empty or invalid weight_map")
        names = sorted(set(weights.values()))
    else:
        names = ["model.safetensors"]
    for name in names:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Expected local safetensors shard filenames")
        if not (source / name).is_file() or not (source / name).stat().st_size:
            raise ValueError("Missing or empty checkpoint shard: " + name)
    # The converter discovers local shards; its inputs must agree with our manifest.
    if {p.name for p in source.glob("*.safetensors")} != set(names):
        raise ValueError("Indexed model files and available safetensors differ")
    return [source / name for name in names], index if index.is_file() else None


def _identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _git_revision(checkout):
    try:
        result = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def export(model, llama_dir, out, *, dry_run=False):
    source, checkout = Path(model).resolve(strict=True), Path(llama_dir).resolve(strict=True)
    target = Path(out).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError("Output already exists: " + str(target))
    if any(root in target.resolve().parents for root in (source, checkout)):
        raise ValueError("Choose a destination outside the checkpoint and converter directories")
    config = source / "config.json"
    settings = json.loads(config.read_text())
    if any("quantization_config" in item for item in (settings, settings.get("text_config") or {})):
        raise ValueError("Use a merged floating-point checkpoint")
    if (source / "adapter_config.json").exists():
        raise ValueError("Merge the adapter into the base checkpoint before exporting")
    weights, index = _local_weights(source)
    converter = checkout / "convert_hf_to_gguf.py"
    quantizer = checkout / "build" / "bin" / "llama-quantize"
    if not converter.is_file() or not quantizer.is_file() or not os.access(quantizer, os.X_OK):
        raise ValueError("Expected a local llama.cpp checkout with built build/bin/llama-quantize")

    # CPU conversion creates a BF16 intermediate and a Q4_K_M export on this filesystem.
    weight_bytes = sum(p.stat().st_size for p in weights)
    estimated_free = (weight_bytes * 14 + 9) // 10
    existing_parent = next(p for p in target.parents if p.exists())
    free = shutil.disk_usage(existing_parent).free
    base_args = [sys.executable, str(converter), str(source), "--outtype", "bf16", "--outfile"]
    if dry_run:
        return {"status": "planned", "source_model": str(source), "out": str(target),
                "estimated_work_bytes": estimated_free, "filesystem_free_bytes": free,
                "convert": base_args + ["<staging>/weights-bf16.gguf"],
                "quantize": [str(quantizer), "<staging>/weights-bf16.gguf",
                             "<staging>/candidate.gguf", "Q4_K_M"]}
    if free < estimated_free:
        raise ValueError(f"Estimated working space {estimated_free} bytes, available {free}")

    target.parent.mkdir(parents=True, exist_ok=True)
    # Retain the lock inode between runs: flock is released even if a process crashes.
    lock_path = target.parent / ("." + target.name + ".export.lock")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another export owns this destination") from exc
        if target.exists() or target.is_symlink():
            raise ValueError("Output already exists: " + str(target))
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".gguf-export-") as directory:
            staging = Path(directory)
            intermediate = staging / "weights-bf16.gguf"
            publish = staging / "ready"
            publish.mkdir()
            candidate = publish / "candidate.gguf"
            # Tokenizer files can influence the GGUF as well. Keep all source file
            # identities stable across hashing and both conversion commands.
            source_files = sorted(p for p in source.iterdir() if p.is_file())
            before = {p: _identity(p) for p in source_files}
            weight_hashes = {p.name: file_hash(p) for p in weights}
            config_hash = file_hash(config)
            index_hash = file_hash(index) if index is not None else None
            tools = {"llama_dir": str(checkout), "git_revision": _git_revision(checkout),
                     "converter_sha256": file_hash(converter), "quantizer_sha256": file_hash(quantizer)}
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            subprocess.run(base_args + [str(intermediate)], check=True, env=env, stdout=sys.stderr)
            subprocess.run([str(quantizer), str(intermediate), str(candidate), "Q4_K_M"],
                           check=True, env=env, stdout=sys.stderr)
            intermediate.unlink()
            artifact = inspect_candidate(candidate, capacity_bytes=B300_NOMINAL_BYTES,
                                         reserve_bytes=DEFAULT_RESERVE_BYTES)
            if before != {p: _identity(p) for p in source.iterdir() if p.is_file()}:
                raise ValueError("Source checkpoint changed during export")
            manifest = {
                "status": "requires_gpu_test", "source_model": str(source),
                "config_sha256": config_hash, "index_sha256": index_hash,
                "source_weight_sha256": weight_hashes, "tools": tools,
                "quantization": "Q4_K_M", "candidate": artifact.to_dict(),
            }
            (publish / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            publish.rename(target)
            return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--llama-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = export(args.model, args.llama_dir, args.out, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
