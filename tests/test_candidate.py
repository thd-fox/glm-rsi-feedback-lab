import struct
import tempfile
import unittest
from pathlib import Path

from b300_candidate import B300_NOMINAL_BYTES, inspect_candidate


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, metadata, names, payload=b"\x00\x00\x00\x00"):
    directory = b""
    for key, value in metadata.items():
        if isinstance(value, str):
            encoded = struct.pack("<I", 8) + string(value)
        elif isinstance(value, list):
            encoded = struct.pack("<IIQ", 9, 8, len(value)) + b"".join(map(string, value))
        else:
            encoded = struct.pack("<Ii", 5, value)
        directory += string(key) + encoded
    for i, name in enumerate(names):
        directory += string(name) + struct.pack("<IQIQ", 1, 1, 0, i * 32)
    data = struct.pack("<4sIQQ", b"GGUF", 3, len(names), len(metadata)) + directory
    path.write_bytes(data.ljust((len(data) + 31) // 32 * 32, b"\0") +
                     payload.ljust(32, b"\0") * len(names))


class CandidateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "test.gguf"
        self.metadata = {"general.architecture": "glm5-next", "general.type": "model",
                         "tokenizer.ggml.tokens": ["a", "b"]}
        write_gguf(self.path, self.metadata, ["token_embd.weight"])

    def inspect(self, path=None, limit=B300_NOMINAL_BYTES, reserve=0):
        return inspect_candidate(path or self.path, capacity_bytes=limit, reserve_bytes=reserve)

    def test_budget_exact_boundary_and_clamp(self):
        size = self.path.stat().st_size
        report = self.inspect(limit=size + 12, reserve=12).to_dict()
        self.assertEqual((report["size_bytes"], report["remaining_bytes"]), (size, 0))
        with self.assertRaises(ValueError):
            self.inspect(limit=size + 11, reserve=12)
        self.assertEqual(self.inspect(limit=2**50).capacity_bytes, B300_NOMINAL_BYTES)

    def test_changed_weight_byte_changes_set_digest(self):
        before = self.inspect()
        write_gguf(self.path, self.metadata, ["token_embd.weight"], payload=b"1")
        self.assertNotEqual(before.sha256, self.inspect().sha256)

    def test_adapters_unrelated_files_and_missing_data_rejected(self):
        for fields, names in [
            ({"general.type": "adapter"}, ["token_embd.weight"]),
            ({"adapter.type": "lora"}, ["token_embd.weight"]),
            ({}, ["blk.0.attn_v.weight.lora_a"]),
            ({}, ["mm.0.weight"]),
            ({"split.count": 2}, ["token_embd.weight"]),
        ]:
            write_gguf(self.path, {**self.metadata, **fields}, names)
            with self.subTest(fields=fields, names=names), self.assertRaises(ValueError):
                self.inspect()
        write_gguf(self.path, self.metadata, ["token_embd.weight"])
        self.path.write_bytes(self.path.read_bytes()[:-32])
        with self.assertRaises(ValueError):
            self.inspect()

    def test_metadata_only_first_shard_and_full_set(self):
        # Real GLM exports can have metadata alone in the first shard.
        first = self.path.with_name("glm-00001-of-00002.gguf")
        second = self.path.with_name("glm-00002-of-00002.gguf")
        split = {"split.count": 2, "split.tensors.count": 1}
        write_gguf(first, {**self.metadata, **split, "split.no": 0}, [])
        with self.assertRaises(FileNotFoundError):
            self.inspect(first)
        write_gguf(second, {**split, "split.no": 1}, ["token_embd.weight"])
        complete = self.inspect(first)
        self.assertEqual(complete.shards, (first, second))
        self.assertEqual(complete.size_bytes, first.stat().st_size + second.stat().st_size)
        with self.assertRaises(ValueError):
            self.inspect(first, limit=first.stat().st_size)
        for changed in [{"split.no": 0}, {"split.count": 3}, {"split.tensors.count": 2}]:
            write_gguf(second, {**split, "split.no": 1, **changed}, ["token_embd.weight"])
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.inspect(first)


if __name__ == "__main__":
    unittest.main()
