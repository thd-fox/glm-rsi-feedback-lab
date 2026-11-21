import argparse
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import b300_fleet as fleet


class FleetTests(unittest.TestCase):
    def test_replica_geometry(self):
        for precision, expected_replicas in (("nvfp4", 4), ("fp8", 2), ("bf16", 2)):
            slots = fleet.plan(precision, None, 8000)["replicas"]
            self.assertEqual(len(slots), expected_replicas)
            self.assertEqual([gpu for slot in slots for gpu in slot["devices"]], list(range(8)))
            self.assertEqual(len({r["port"] for r in slots}), expected_replicas)
        with self.assertRaises(ValueError):
            fleet.plan("fp8", 4, 8000)
        with self.assertRaises(ValueError):
            fleet.plan("nvfp4", None, 65535)

    def test_create_uses_pinned_profile_and_localhost(self):
        tools = fleet.plan("nvfp4", 1, 8200)
        with patch.object(fleet.subprocess, "check_output", return_value="id\n") as runner:
            self.assertEqual(fleet.create_replica("test", 0, ["GPU-a", "GPU-b"], 8200,
                                                  tools["profile"], "sha256:test", Path("/weights"), None), "id")
        command = runner.call_args.args[0]
        for value in ('"device=GPU-a,GPU-b"', "127.0.0.1:8200:8000",
                      "NIM_MODEL_PROFILE=" + tools["profile"],
                      "type=bind,src=/weights,dst=/weights/snapshots/base,readonly"):
            self.assertIn(value, command)

    def test_failed_readiness_rolls_back_all_own_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(name="test", model=Path(tmp), precision="nvfp4", timeout=1)
            ids = [str(i) for i in range(4)]
            devices = ["GPU-" + str(i) for i in range(8)]
            with contextlib.ExitStack() as stack:
                for item, value in (("validate_checkpoint", None), ("reserve_devices", contextlib.nullcontext(devices)),
                                    ("containers", []), ("fleet_lock", contextlib.nullcontext())):
                    stack.enter_context(patch.object(fleet, item, return_value=value))
                stack.enter_context(patch.object(fleet.subprocess, "check_output", return_value="sha256:"+"a"*64))
                stack.enter_context(patch.object(fleet.subprocess, "run"))
                create = stack.enter_context(patch.object(fleet, "create_replica", side_effect=ids))
                stack.enter_context(patch.object(fleet, "wait_ready", side_effect=RuntimeError("timeout")))
                remove = stack.enter_context(patch.object(fleet, "remove"))
                with self.assertRaisesRegex(RuntimeError, "timeout"):
                    fleet.start(args, fleet.plan("nvfp4", 4, 8100))
                self.assertEqual([c.args[2] for c in create.call_args_list],
                                 [devices[i:i+2] for i in range(0, 8, 2)])
                remove.assert_called_once_with(ids)

    def test_mismatched_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
                "text_config": {"dtype": "bfloat16"}}))
            with self.assertRaisesRegex(ValueError, "nvfp4"):
                fleet.validate_checkpoint(root, "nvfp4")
