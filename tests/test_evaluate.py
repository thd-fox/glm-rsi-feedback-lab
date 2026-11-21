from http.server import BaseHTTPRequestHandler, HTTPServer
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from b300_evaluate import evaluate, evaluate_to_report


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.checks = root / "holdout"
        self.training = root / "training"
        self.report = root / "report"
        self.training.write_text(json.dumps({"messages": [
            {"role": "user", "content": "train-only"}, {"role": "assistant", "content": "{}"}]}) + "\n")
        self.cases = [
            {"case_id": "case-a", "group_id": "group-a", "kind": "target",
             "prompt": "hidden target", "expected_json": {"hidden_target_key": True}},
            {"case_id": "case-b", "group_id": "group-b", "kind": "control",
             "prompt": "hidden control", "expected_json": {"hidden_control_key": False}},
        ]
        self.store_cases()

    def store_cases(self):
        self.checks.write_text("".join(json.dumps(c) + "\n" for c in self.cases))

    def arguments(self, ports=(8080, 8081)):
        return (*ports, "1" * 64, "2" * 64, self.checks, self.training)

    def test_paired_requests_and_auditable_passing_report(self):
        records, ports = [], []
        cases = self.cases

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                case = next(c for c in cases if c["prompt"] == json.loads(payload)["messages"][0]["content"])
                records.append((self.server.server_port, payload))
                correct = self.server.server_port == ports[1] or case["kind"] == "control"
                choice = {"finish_reason": "stop", "message": {
                    "content": json.dumps(case["expected_json"]) if correct else "{}",
                    "reasoning_content": "secret thinking"}}
                body = json.dumps({"model": "glm-5.3-flash", "choices": [choice]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        for _ in range(2):
            server = HTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            ports.append(server.server_port)
        result = evaluate_to_report(self.report, *self.arguments(ports))
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["target_fixes"], result["controls_passed"], result["regressions"]),
                         (1, 1, 0))
        self.assertEqual(json.loads(self.report.read_text()), result)
        self.assertEqual(result["checks_sha256"], hashlib.sha256(self.checks.read_bytes()).hexdigest())
        self.assertEqual(result["training_data_sha256"], hashlib.sha256(self.training.read_bytes()).hexdigest())
        self.assertEqual([b for p, b in records if p == ports[0]], [b for p, b in records if p == ports[1]])
        for _, req in records:
            self.assertNotIn(b"hidden_target_key", req)
        for raw in ("hidden target", "hidden_control_key", "group-a", "secret thinking"):
            self.assertNotIn(raw, self.report.read_text())
        with self.assertRaises(ValueError):
            evaluate_to_report(self.report, *self.arguments(ports))

    def test_control_regression_and_no_improvement_reject(self):
        def answers(port, body):
            target = json.loads(body)["messages"][0]["content"] == "hidden target"
            if target:
                return '{"hidden_target_key":true}' if port == 8081 else "{}"
            return '{"hidden_control_key":false}' if port == 8080 else '{"hidden_control_key":0}'

        result = evaluate_to_report(self.report, *self.arguments(), request=answers)
        self.assertEqual((result["status"], result["regressions"], result["target_fixes"]),
                         ("rejected", 1, 1))
        result = evaluate(*self.arguments(), request=lambda *args: None)
        self.assertEqual((result["status"], result["target_fixes"]), ("rejected", 0))

    def test_overlap_and_ambiguous_identity_block_before_requests(self):
        def never(*_):
            self.fail("request reached the model")

        with self.assertRaises(ValueError):
            evaluate(*self.arguments((8000, 8000)), request=never)
        self.cases[0]["prompt"] = " TRAIN-ONLY  "
        self.store_cases()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            evaluate_to_report(self.report, *self.arguments(), request=never)
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
