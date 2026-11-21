"""Content identity of local Hugging Face checkpoint shards, without loading tensors."""
import hashlib
import os
from pathlib import Path


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def stable_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        before = _identity(os.fstat(source.fileno()))
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
        if before != _identity(os.fstat(source.fileno())) or before != _identity(path.stat()):
            raise ValueError(f"Source file changed while hashing: {path.name}")
    return digest.hexdigest()


class WeightSnapshot:
    """Check cheap file identities again after a long training or transformation job."""
    def __init__(self, model):
        self.model = model
        self.initial = self._files()

    def _files(self):
        return {p.name: _identity(p.stat()) for p in self.model.iterdir() if p.is_file()}

    def check(self):
        if self._files() != self.initial:
            raise ValueError("Source checkpoint changed during the operation")

    def weight_hashes(self, index):
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Missing weight map in source checkpoint")
        if any(not isinstance(n, str) or Path(n).name != n or not n.endswith(".safetensors")
               or n not in self.initial for n in weights.values()):
            raise ValueError("Invalid or missing indexed source shard")
        hashes = {name: stable_digest(self.model / name) for name in sorted(set(weights.values()))}
        self.check()
        return hashes

    def verify_weights(self, index, manifest):
        expected = manifest.get("source_weight_sha256")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Missing source_weight_sha256 in manifest; original weight bytes are unverified")
        if self.weight_hashes(index) != expected:
            raise ValueError("Source weight SHA-256 mismatch with manifest")
