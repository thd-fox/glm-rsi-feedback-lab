import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None

from b300_checkpoint import INDEX, digest, grow, merge
from b300_lineage import WeightSnapshot


@unittest.skipIf(torch is None, "install the checkpoint extra")
class CheckpointTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.model = self.root / "base"
        self.model.mkdir()
        self.run = self.root / "run"
        adapter = self.run / "artifacts" / "adapter"
        adapter.mkdir(parents=True)
        self.export = self.root / "export"
        self.export.mkdir()
        self.fit = self.root / "fit.json"
        self.out = self.root / "result"
        self.prefix = "model.language_model.layers.0."
        self.key = self.prefix + "mlp.down_proj.weight"
        rng = torch.Generator().manual_seed(42)
        self.weights = {self.prefix + "mlp." + name + ".weight":
                        torch.randn(shape, generator=rng).bfloat16() for name, shape in
                        (("up_proj", (256, 4)), ("gate_proj", (256, 4)), ("down_proj", (4, 256)))}
        save_file(self.weights, str(self.model / "part-1.safetensors"))
        save_file({"untouched": torch.ones(1)}, str(self.model / "part-2.safetensors"))
        self.write(self.model / INDEX, {"metadata": {"total_size": 6148},
                   "weight_map": {**dict.fromkeys(self.weights, "part-1.safetensors"),
                                  "untouched": "part-2.safetensors"}})
        self.write(self.model / "config.json", {"model_type": "glm5_next", "text_config": {
            "dtype": "bfloat16", "intermediate_size": 256, "hidden_size": 4,
            "first_k_dense_replace": 1, "mlp_layer_types": ["dense"]}})
        shard_hashes = WeightSnapshot(self.model).weight_hashes(json.loads((self.model / INDEX).read_text()))
        self.write(self.run / "run.json", {"status": "trained", "base_model": str(self.model),
                   "config_sha256": digest(self.model / "config.json"),
                   "source_weight_sha256": shard_hashes,
                   "weight_index_sha256": digest(self.model / INDEX)})
        self.write(adapter / "adapter_config.json", {"peft_type": "LORA", "r": 2, "lora_alpha": 2})
        self.a, self.b = torch.ones(2, 256), torch.ones(4, 2)
        target = "base_model.model." + self.key.removesuffix(".weight")
        save_file({target + ".lora_A.weight": self.a, target + ".lora_B.weight": self.b},
                  str(adapter / "adapter_model.safetensors"))
        identity = {"sha256": "abc", "size_bytes": 1048576, "reserve_bytes": 0}
        self.write(self.export / "export-manifest.json", {"candidate": identity,
                   "source_model": str(self.model), "config_sha256": digest(self.model / "config.json"),
                   "source_weight_sha256": shard_hashes,
                   "index_sha256": digest(self.model / INDEX)})
        self.report = {"status": "passed", "candidate": identity, "model_sha256": "abc",
                       "parallel_slots": 1, "gpu": {"total_mib": 10},
                       "sampled_device_peak_mib_including_reservation": 6}
        self.write(self.fit, self.report)

    def write(self, path, obj):
        path.write_text(json.dumps(obj))

    def test_merge_updates_only_selected_bf16_tensor(self):
        merge(self.model, self.run, self.out)
        after = load_file(str(self.out / "part-1.safetensors"))
        self.assertTrue(torch.equal(after[self.key], (self.weights[self.key].float() + self.b @ self.a).bfloat16()))
        self.assertEqual((self.out / "part-2.safetensors").read_bytes(), (self.model / "part-2.safetensors").read_bytes())
        self.assertTrue(torch.equal(load_file(str(self.model / "part-1.safetensors"))[self.key], self.weights[self.key]))
        with self.assertRaises(FileExistsError):
            merge(self.model, self.run, self.out)

    def test_growth_adds_trainable_dimensions_preserving_output(self):
        grow(self.model, self.export, self.fit, self.out, 256, 17)
        after = load_file(str(self.out / "part-1.safetensors"))
        x = torch.tensor([1., -2., 0., 0.5])
        def mlp(tensors):
            weight = lambda name: tensors[self.prefix + "mlp." + name + ".weight"].float()
            return weight("down_proj") @ (torch.nn.functional.silu(weight("gate_proj") @ x) * (weight("up_proj") @ x))
        torch.testing.assert_close(mlp(after), mlp(self.weights))
        self.assertEqual(after[self.key].shape, (4, 512))
        self.assertEqual(json.loads((self.out / INDEX).read_text())["metadata"]["total_size"], 6148 + 3*4*256*2)

    def test_growth_rejects_mismatch_and_exhausted_memory(self):
        for changes in ({"status": "failed"}, {"model_sha256": "other"},
                        {"sampled_device_peak_mib_including_reservation": 10}):
            self.write(self.fit, {**self.report, **changes})
            with self.assertRaises(ValueError):
                grow(self.model, self.export, self.fit, self.out, 256, 17)
        self.assertFalse(self.out.exists())

    def test_merge_and_grow_reject_same_index_with_changed_shard(self):
        save_file({"untouched": torch.zeros(1)}, str(self.model / "part-2.safetensors"))
        with self.assertRaisesRegex(ValueError, "weight SHA-256 mismatch"):
            merge(self.model, self.run, self.out)
        with self.assertRaisesRegex(ValueError, "weight SHA-256 mismatch"):
            grow(self.model, self.export, self.fit, self.out, 256, 17)
        self.assertFalse(self.out.exists())

    def test_byte_identical_relocated_base(self):
        relocated = self.model.rename(self.root / "moved")
        merge(relocated, self.run, self.out)
        grow(relocated, self.export, self.fit, self.root / "grown", 256, 17)
