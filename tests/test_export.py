import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from b300_export import export


def tiny_model():
    def string(s):
        return struct.pack("<Q", len(s)) + s

    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 1)
    header += string(b"general.architecture") + struct.pack("<I", 8) + string(b"glm5-next")
    header += string(b"token_embd.weight") + struct.pack("<IQIQ", 1, 1, 0, 0)
    return header.ljust((len(header) + 31) // 32 * 32, b"\0") + struct.pack("<f", 1)


class ExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.source = root / "merged"
        self.source.mkdir()
        (self.source / "config.json").write_text('{"model_type":"fixture"}')
        weight_name = "model-00001-of-00001.safetensors"
        (self.source / weight_name).write_bytes(b"weight fixture")
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": weight_name}}))
        self.checkout = root / "llama"
        binary = self.checkout / "build" / "bin"
        binary.mkdir(parents=True)
        self.quant = binary / "llama-quantize"
        self.quant.write_text(
            '#!/bin/sh\n[ "$3" = Q4_K_M ] && [ "$HF_HUB_OFFLINE" = 1 ] && cp "$1" "$2"\n')
        self.quant.chmod(0o755)
        (self.checkout / "convert_hf_to_gguf.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "assert sys.argv[2:5] == ['--outtype', 'bf16', '--outfile']\n"
            f"Path(sys.argv[-1]).write_bytes({tiny_model()!r})\n")
        self.out = root / "new" / "export"

    def test_dry_run_creates_nothing(self):
        result = export(self.source, self.checkout, self.out, dry_run=True)
        self.assertEqual(result["status"], "planned")
        self.assertFalse(self.out.parent.exists())

    def test_publish_complete_manifest_and_prevent_overwrite(self):
        result = export(self.source, self.checkout, self.out)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["candidate.gguf", "export-manifest.json"])
        self.assertEqual(result, json.loads((self.out / "export-manifest.json").read_text()))
        self.assertEqual(result["source_model"], str(self.source.resolve()))
        self.assertEqual(result["index_sha256"], hashlib.sha256(
            (self.source / "model.safetensors.index.json").read_bytes()).hexdigest())
        self.assertEqual(result["candidate"]["shards"][0]["sha256"],
                         hashlib.sha256(tiny_model()).hexdigest())
        with self.assertRaises(ValueError):
            export(self.source, self.checkout, self.out)

    def test_partial_failure_is_clean_and_retryable(self):
        good = self.quant.read_text()
        self.quant.write_text('#!/bin/sh\nprintf partial > "$2"\nexit 1\n')
        with self.assertRaises(subprocess.CalledProcessError):
            export(self.source, self.checkout, self.out)
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.out.parent.glob(".gguf-export-*")))
        self.quant.write_text(good)
        self.assertEqual(export(self.source, self.checkout, self.out)["status"], "requires_gpu_test")

    def test_reject_incomplete_source_and_nested_destination(self):
        with self.assertRaises(ValueError):
            export(self.source, self.checkout, self.source / "nested", dry_run=True)
        (self.source / "model-00001-of-00001.safetensors").unlink()
        with self.assertRaises(ValueError):
            export(self.source, self.checkout, self.out, dry_run=True)


if __name__ == "__main__":
    unittest.main()
