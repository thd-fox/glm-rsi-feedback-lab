import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import b300_rotation as rotation
from b300_router import leased_port


class RotationTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.state = Path(workspace.name)
        self.groups = {"a": {"port": 8100}, "b": {"port": 8101}}
        self.pointer("a", "first")

    def pointer(self, slot, publication):
        rotation.publish(self.state / "current.json", {
            "active_group": slot, "groups": self.groups, "publication": publication,
        })

    def journal(self, old, new, previous, attempt):
        rotation.publish(self.state / "pending.json", {
            "old": old, "target": new, "groups": self.groups, "parent": previous, "attempt": attempt,
        })

    def test_two_promotions_route_to_opposite_groups_and_drain_old_reader(self):
        for old, new, previous, attempt in (("a","b","first","second"), ("b","a","second","third")):
            self.journal(old, new, previous, attempt)
            with leased_port(self.state, 8100) as existing:
                self.pointer(new, attempt)
                with leased_port(self.state, 8100) as incoming:
                    self.assertEqual(existing, self.groups[old]["port"])
                    self.assertEqual(incoming, self.groups[new]["port"])
                with (self.state / f"slot-{old}.lease").open("a") as retire:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(retire, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(rotation, "stop_slot") as stop:
                rotation.recover_pending(self.state, 0)
                stop.assert_called_once_with(self.state, old, self.groups[old], 0)
            self.assertFalse((self.state / "pending.json").exists())

    def test_rejected_or_interrupted_round_only_retires_standby(self):
        self.journal("a", "b", "first", "rejected")
        with patch.object(rotation, "stop_slot") as stop:
            rotation.recover_pending(self.state, 0)
            stop.assert_called_once_with(self.state, "b", self.groups["b"], 0)
        with leased_port(self.state, 8100) as active:
            self.assertEqual(active, 8100)
