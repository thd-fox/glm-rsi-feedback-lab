from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from b300_feedback import prepare_batch


class FeedbackBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cases, self.oracles, self.out = (self.root / s for s in ["cases", "oracles", "out"])

    def case(self, i, **kwargs):
        return {"case_id": f"id-{i}", "group_id": f"group-{i}", "split": "train",
                "training_allowed": True, "prompt": f"task {i}", "rejected": "wrong", **kwargs}

    def write(self, cases, expected):
        def rows(path, values):
            path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
        rows(self.cases, cases)
        rows(self.oracles, [{"case_id": k, "expected_json": v} for k, v in expected.items()])

    def records(self, file):
        return [json.loads(l) for l in (self.out / file).read_text().splitlines()]

    def test_four_local_replicas_order_secrecy_and_audit(self):
        barrier = threading.Barrier(4)
        received, ports = [], []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.server.server_port, self.path, body))
                barrier.wait(timeout=4)
                result = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "reasoning_content": "hidden", "content": '{"secret_oracle_key": true}'}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

            def log_message(self, *args):
                pass

        for _ in range(4):
            server = HTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                                      daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            ports.append(server.server_port)
        cases = [self.case(i) for i in range(4)]
        self.write(cases, {c["case_id"]: {"secret_oracle_key": True} for c in cases})
        result = prepare_batch(self.cases, self.oracles, self.out, ports)
        self.assertEqual(result["accepted"], 4)
        self.assertEqual({p for p, _, _ in received}, set(ports))
        for _, path, body in received:
            self.assertEqual((path, body["model"]), ("/v1/chat/completions", "glm-5.3-flash"))
            self.assertNotIn("secret_oracle_key", json.dumps(body))
        rows = self.records("train.jsonl")
        self.assertEqual([row["messages"][0]["content"] for row in rows], [c["prompt"] for c in cases])
        self.assertNotIn("hidden", json.dumps(rows))
        for item in self.records("audit.jsonl"):
            self.assertEqual(set(item), {"case_id_sha256", "group_id_sha256", "reason", "source"})
            self.assertEqual(item["reason"], "accepted")
        with self.assertRaises(ValueError):
            prepare_batch(self.cases, self.oracles, self.out, ports)

    def test_filter_and_exact_json_semantics(self):
        cases = [self.case(0, correction='{"n": 1, "flag":true}'),
                 self.case(1, training_allowed=False), self.case(2, split="test"),
                 self.case(3), self.case(4, rejected='{"flag":true,"n":1.00}'),
                 self.case(5, correction='{"flag":1,"n":true}')]
        expected = {c["case_id"]: {"flag": True, "n": 1} for c in cases if c["case_id"] != "id-3"}
        self.write(cases, expected)
        called = []

        def generate(port, case, attempt):
            called.append((port, case.case_id, attempt))
            return ['{"flag":true,"n":1,"n":1}', '{"flag":true,"n":NaN}', '{"n":1,"flag":true}'][attempt-1]

        result = prepare_batch(self.cases, self.oracles, self.out, [8000], generator=generate)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(called, [(8000, "id-5", n) for n in (1, 2, 3)])
        self.assertEqual([a["reason"] for a in self.records("audit.jsonl")],
                         ["accepted", "training_disallowed", "outside_train", "original_unverified",
                          "original_passed", "accepted"])

    def test_leakage_checked_before_generation(self):
        for overlap in [{"group_id": "GROUP-0"}, {"prompt": " Task    0 "}, {"case_id": "id-0"}]:
            cases = [self.case(0), self.case(1, split="test", **overlap)]
            self.write(cases, {"id-0": {}})
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                prepare_batch(self.cases, self.oracles, self.out,
                              generator=lambda *_: self.fail("premature call"))
            self.assertFalse(self.out.exists())

    def test_generation_exception_discards_whole_batch(self):
        self.write([self.case(0, correction="{}"), self.case(1)], {"id-0": {}, "id-1": {}})

        def fail(*_):
            raise OSError("replica offline")

        with self.assertRaises(OSError):
            prepare_batch(self.cases, self.oracles, self.out, generator=fail)
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob(".feedback-*")))


if __name__ == "__main__":
    unittest.main()
