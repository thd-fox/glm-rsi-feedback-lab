import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import b300_train as training


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.blobs = self.root / "hub/blobs"
        self.model = self.root / "hub/snapshots/revision"
        self.model.mkdir(parents=True)
        self.blobs.mkdir()
        (self.blobs / "shard").write_bytes(b"stub")
        (self.model / "a.safetensors").symlink_to("../../blobs/shard")
        for name, obj in {
            "config.json": {"model_type": "glm5_next", "text_config": {"dtype": "bfloat16"}},
            "model.safetensors.index.json": {"weight_map": {"a": "a.safetensors"}},
            "tokenizer_config.json": {}, "tokenizer.json": {},
        }.items():
            (self.model / name).write_text(json.dumps(obj))
        self.data = self.root / "source.jsonl"
        self.data.write_text(json.dumps({"feedback": "private note", "messages": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}) + "\n")
        self.run_dir = self.root / "run"
        self.args = argparse.Namespace(model=self.model, data=self.data, out=self.run_dir,
                                       gpus=8, steps=1, image="local", dry_run=False)

    def test_dry_run_then_isolated_launch(self):
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()):
            training.train(self.args)
        self.assertFalse(self.run_dir.exists())
        self.args.dry_run = False
        devices = [f"GPU-{n}" for n in range(8)]

        def launch(cmd, check):
            self.assertTrue(check)
            record = json.loads((self.run_dir / "run.json").read_text())
            self.assertEqual(record["source_weight_sha256"],
                             {"a.safetensors": hashlib.sha256(b"stub").hexdigest()})
            self.assertIn("--nproc-per-node=8", cmd)
            self.assertIn("--network=none", cmd)
            self.assertIn(f"type=bind,src={self.blobs},dst=/weights/blobs,readonly", cmd)
            self.assertIn('"device=' + ",".join(devices) + '"', cmd)
            adapter = self.run_dir / "artifacts/adapter"
            adapter.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (adapter / name).write_bytes(b"stub")

        with patch.object(training, "reserve_devices", return_value=contextlib.nullcontext(devices)), \
                patch.object(training.subprocess, "check_output", return_value="sha256:" + "0"*64), \
                patch.object(training.subprocess, "run", side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
            training.train(self.args)
        exported = (self.run_dir / "inputs/train.jsonl").read_bytes()
        manifest = json.loads((self.run_dir / "run.json").read_text())
        self.assertEqual(manifest["training_data_sha256"], hashlib.sha256(exported).hexdigest())
        self.assertEqual(manifest["status"], "trained")
        self.assertNotIn(b"private note", exported)
        with self.assertRaises(ValueError):
            training.train(self.args)

    def test_missing_weight_or_invalid_sft_stops_dry_run(self):
        self.args.dry_run = True
        self.data.write_text('{"messages":[]}\n')
        with self.assertRaisesRegex(ValueError, "Line 1"):
            training.train(self.args)
        (self.blobs / "shard").unlink()
        with self.assertRaisesRegex(ValueError, "Missing model file"):
            training.train(self.args)
        self.assertFalse(self.run_dir.exists())

    def test_allocators_get_disjoint_cards_and_release_on_failure(self):
        devices = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
        with patch.object(training, "available_devices", return_value=devices):
            with training.reserve_devices(2, directory=self.root) as first:
                self.assertEqual(first, devices[:2])
                with training.reserve_devices(2, directory=self.root) as second:
                    self.assertEqual(second, devices[2:])
                    with self.assertRaisesRegex(ValueError, "could be reserved"):
                        with training.reserve_devices(1, directory=self.root):
                            self.fail("overlapping reservation")
                with self.assertRaises(KeyboardInterrupt):
                    with training.reserve_devices(2, directory=self.root):
                        raise KeyboardInterrupt
            with training.reserve_devices(4, directory=self.root) as all_cards:
                self.assertEqual(all_cards, devices)

    def test_parent_host_changes_source_during_training(self):
        def launch(cmd, check):
            (self.blobs / "shard").write_bytes(b"edit")
            adapter = self.run_dir / "artifacts/adapter"
            adapter.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (adapter / name).write_bytes(b"stub")

        with patch.object(training, "reserve_devices", return_value=contextlib.nullcontext(["GPU-1", "GPU-2"])), \
                patch.object(training.subprocess, "check_output", return_value="sha256:" + "0"*64), \
                patch.object(training.subprocess, "run", side_effect=launch):
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                training.train(self.args)
        self.assertEqual(json.loads((self.run_dir / "run.json").read_text())["status"], "failed")
