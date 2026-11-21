"""Weigh and fingerprint a complete, merged text-model GGUF before a GPU fit test."""
import argparse
import hashlib
import json
import os
import re
import stat
import struct
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

B300_NOMINAL_BYTES = 288_000_000_000
DEFAULT_RESERVE_BYTES = 8 * 1024 ** 3
_SPLIT = re.compile(r"(.+)-([0-9]{5})-of-([0-9]{5})\.gguf")
# GGUF v3 metadata scalar types in enum order. Strings and arrays are variable-length.
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
_FIELDS = {"general.architecture", "general.type", "adapter.type", "general.alignment",
           "split.no", "split.count", "split.tensors.count"}


@dataclass(frozen=True)
class Candidate:
    entrypoint: Path
    shards: tuple
    sizes: tuple
    hashes: tuple
    sha256: str
    capacity_bytes: int
    reserve_bytes: int

    @property
    def size_bytes(self):
        return sum(self.sizes)

    def to_dict(self):
        return {
            "sha256": self.sha256,
            "entrypoint": self.entrypoint.name,
            "shards": [{"name": p.name, "size_bytes": size, "sha256": digest}
                       for p, size, digest in zip(self.shards, self.sizes, self.hashes)],
            "size_bytes": self.size_bytes,
            "capacity_bytes": self.capacity_bytes,
            "reserve_bytes": self.reserve_bytes,
            "remaining_bytes": self.capacity_bytes - self.reserve_bytes - self.size_bytes,
        }


class _Header:
    """Read little-endian GGUF v3 directories without materializing token arrays or weights."""

    def __init__(self, stream, size):
        self.file = stream
        self.size = size

    def read(self, count):
        data = self.file.read(count)
        if len(data) != count:
            raise ValueError("Truncated GGUF header")
        return data

    def skip(self, count):
        if count > self.size - self.file.tell():
            raise ValueError("GGUF value exceeds file length")
        self.file.seek(count, os.SEEK_CUR)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep):
        length = self.number("Q")
        if keep:
            if length > 65536:
                raise ValueError("Invalid GGUF identifier length")
            return self.read(length).decode("utf-8")
        self.skip(length)

    def value(self, kind, keep):
        if kind in _SCALAR:
            return self.number(_SCALAR[kind])
        if kind == 8:
            return self.string(keep)
        if kind != 9 or keep:
            raise ValueError("Unexpected GGUF metadata type")
        item = self.number("I")
        count = self.number("Q")
        if item in _SCALAR:
            self.skip(count * struct.calcsize("<" + _SCALAR[item]))
        elif item == 8 and count <= (self.size - self.file.tell()) // 8:
            for _ in range(count):
                self.string(False)
        else:
            raise ValueError("Invalid or unsupported GGUF array")

    def inspect(self, names):
        if self.read(4) != b"GGUF" or self.number("I") != 3:
            raise ValueError("Expected a little-endian GGUF v3 model")
        tensors, fields = self.number("Q"), self.number("Q")
        metadata = {}
        for _ in range(fields):
            key = self.string(True)
            value = self.value(self.number("I"), key in _FIELDS)
            if key in _FIELDS:
                if key in metadata:
                    raise ValueError("Duplicate GGUF metadata: " + key)
                metadata[key] = value
        last_offset = 0
        for _ in range(tensors):
            name = self.string(True)
            if name in names:
                raise ValueError("Duplicate model tensor: " + name)
            if name.endswith((".lora_a", ".lora_b")):
                raise ValueError("Merge adapter tensors into the base model before export")
            names.add(name)
            dims = self.number("I")
            if not 1 <= dims <= 4:
                raise ValueError("Invalid GGUF tensor shape")
            self.skip(8 * dims + 4)  # shape and GGML tensor type; backend validates payloads
            last_offset = max(last_offset, self.number("Q"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF tensor alignment")
        data_start = (self.file.tell() + alignment - 1) // alignment * alignment
        if tensors and data_start + last_offset >= self.size:
            raise ValueError("Missing GGUF tensor data")
        return metadata


def _shards(entry):
    if entry.suffix != ".gguf":
        raise ValueError("Supply a merged GGUF model or the first GGUF shard")
    match = _SPLIT.fullmatch(entry.name)
    if not match:
        return (entry,)
    prefix, first, total = match.groups()
    if first != "00001" or int(total) == 0:
        raise ValueError("Pass the first shard of a GGUF split")
    return tuple(entry.with_name(f"{prefix}-{i:05d}-of-{total}.gguf")
                 for i in range(1, int(total) + 1))


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def inspect_candidate(model, *, capacity_bytes: int, reserve_bytes: int) -> Candidate:
    """Check the combined byte limit and fingerprint every GGUF shard in inference order.

    This preliminary check is for text inference with all adapters merged. A separate
    load and workload measurement must verify format, GPU placement and runtime memory.
    """
    if type(capacity_bytes) is not int or type(reserve_bytes) is not int:
        raise ValueError("Expected integer byte limits")
    capacity = min(B300_NOMINAL_BYTES, capacity_bytes)
    if not 0 <= reserve_bytes < capacity:
        raise ValueError("Expected 0 <= reserve < single-card capacity")
    entry = Path(model).absolute()
    parts = _shards(entry)
    snapshots = [p.stat() for p in parts]
    if any(not stat.S_ISREG(s.st_mode) for s in snapshots):
        raise ValueError("Every GGUF shard must be a regular file")
    sizes = tuple(s.st_size for s in snapshots)
    if sum(sizes) > capacity - reserve_bytes:
        raise ValueError(f"Candidate is {sum(sizes)} bytes; single-card budget is "
                         f"{capacity - reserve_bytes} bytes after reserve")
    names = set()
    hashes = []
    expected_tensors: Optional[int] = None
    arch = None
    for index, (part, snapshot) in enumerate(zip(parts, snapshots)):
        with part.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(snapshot):
                raise ValueError("Candidate changed during inspection")
            metadata = _Header(stream, snapshot.st_size).inspect(names)
            if index == 0:
                arch = metadata.get("general.architecture")
                if not isinstance(arch, str) or not arch or arch in ("clip", "mmproj"):
                    raise ValueError("First shard must describe a text-model architecture")
                expected_tensors = metadata.get("split.tensors.count")
            if (metadata.get("general.type", "model") != "model" or "adapter.type" in metadata
                    or metadata.get("general.architecture", arch) != arch):
                raise ValueError("Expected a consistent, merged text model")
            split = metadata.get("split.count", 1)
            if type(split) is not int or split not in ((0, 1) if len(parts) == 1 else (len(parts),)):
                raise ValueError("Filename and GGUF split count disagree")
            if len(parts) > 1 and (metadata.get("split.no") != index
                                   or type(expected_tensors) is not int
                                   or metadata.get("split.tensors.count") != expected_tensors):
                raise ValueError("Incomplete or mismatched GGUF shards")
            if len(parts) == 1 and metadata.get("split.no", 0) != 0:
                raise ValueError("Not a standalone GGUF")
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
            hashes.append(digest.hexdigest())
    if "token_embd.weight" not in names:
        raise ValueError("Missing base-model token embeddings")
    if expected_tensors is not None and expected_tensors != len(names):
        raise ValueError("GGUF split tensor count differs from observed tensors")
    if any(_identity(p.stat()) != _identity(s) for p, s in zip(parts, snapshots)):
        raise ValueError("Candidate changed during inspection")
    manifest = [(p.name, size, digest) for p, size, digest in zip(parts, sizes, hashes)]
    sha256 = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    return Candidate(entry, parts, sizes, tuple(hashes), sha256, capacity, reserve_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--capacity-gib", type=Decimal, default=None)
    parser.add_argument("--reserve-gib", type=Decimal, default=Decimal(8))
    args = parser.parse_args()
    try:
        capacity = (B300_NOMINAL_BYTES if args.capacity_gib is None
                    else int(args.capacity_gib * (1 << 30)))
        candidate = inspect_candidate(args.model, capacity_bytes=capacity,
                                      reserve_bytes=int(args.reserve_gib * (1 << 30)))
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Rejected: {error}\n")
    print(json.dumps({"status": "requires_gpu_test", **candidate.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
