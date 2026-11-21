import io
import json
import socket
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import b300_fit as fit


class FitTests(unittest.TestCase):
    def test_cpu_expert_weight_or_cache_rejects_even_with_all_layers(self):
        log = ("load_tensors: offloaded 46/46 layers to GPU\n"
               "load_tensors: CUDA0 model buffer size = 200.00 MiB\n"
               "llama_kv_cache: CUDA0 KV buffer size = 1.50 MiB\n")
        self.assertEqual(fit.require_placement(log)["layers_on_gpu"], 46)
        for problem in ("CPU_Mapped model buffer size = 1.56 MiB", "CPU RS buffer size = 1.12 MiB",
                        "CUDA1 KV buffer size = 2.00 MiB", "offloaded 45/46 layers to GPU"):
            with self.subTest(problem=problem), self.assertRaises(RuntimeError):
                fit.require_placement(log + problem)
        with self.assertRaises(RuntimeError):
            fit.require_placement("offloaded 46/46 layers to GPU")

    def test_full_context_completion_checks_counts_even_when_decode_stops(self):
        def reply(path, endpoint, body, deadline):
            if endpoint == "/tokenize":
                return {"tokens": [31, 52]}
            self.assertEqual(len(body["prompt"]), 96)
            self.assertFalse(body["cache_prompt"])
            self.assertEqual(body["n_predict"], 32)
            return {"tokens_evaluated": 96, "tokens_predicted": 32, "stop_type": "limit"}

        with patch.object(fit, "request", side_effect=reply):
            self.assertEqual(fit.complete(None, 128, 0)["generated_tokens"], 32)
        with patch.object(fit, "request", side_effect=[
            {"tokens": [31]}, {"tokens_evaluated": 96, "tokens_predicted": 3, "stop_type": "eos"},
        ]), self.assertRaises(RuntimeError):
            fit.complete(None, 128, 0)

    def test_report_cannot_replace_existing_and_failed_fit_exposes_no_socket(self):
        @contextmanager
        def failed(args):
            raise RuntimeError("GPU capacity exceeded")
            yield

        with TemporaryDirectory() as tempdir:
            directory = Path(tempdir)
            report = directory / "result.json"
            sock = directory / "model.sock"
            args = ["--model", "/dev/null", "--serve", "--socket", str(sock), "--report", str(report)]
            with patch.object(fit, "measure", failed), redirect_stderr(io.StringIO()):
                self.assertEqual(fit.main(args), 1)
            self.assertEqual(json.loads(report.read_text())["status"], "failed")
            self.assertFalse(sock.is_symlink())
            with self.assertRaises(FileExistsError):
                fit.publish(report, {"status": "passed"})

    def test_only_one_physical_b300(self):
        valid = "GPU-11111111-1111-1111-1111-111111111111, NVIDIA B300, 275000, 192, Disabled, 590"
        with patch.object(fit.subprocess, "run") as run:
            run.return_value.stdout = valid
            self.assertEqual(fit.gpu_info("0")["total_mib"], 275000)
            for invalid in (valid.replace("B300", "B200"), valid.replace("Disabled", "Enabled"),
                            valid + "\n" + valid):
                run.return_value.stdout = invalid
                with self.assertRaises(ValueError):
                    fit.gpu_info("0")

    def test_serve_exposes_the_checked_process_via_private_unix_socket(self):
        with TemporaryDirectory(dir="/tmp") as tmp:
            directory = Path(tmp)
            link = directory / "candidate.sock"
            original = directory / "measured.sock"
            report = directory / "report.json"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(original))
                server.listen()

                def serve_request():
                    connection, _ = server.accept()
                    with connection:
                        connection.recv(2048)
                        connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{"status":"ok"}')

                class FakeProcess:
                    def wait(inner_self):
                        self.assertTrue(link.is_symlink())
                        self.assertEqual(json.loads(report.read_text())["status"], "passed")
                        self.assertEqual(fit.request(link, "/health", None, time.monotonic() + 3),
                                         {"status": "ok"})
                        raise KeyboardInterrupt()

                @contextmanager
                def measured(args):
                    self.assertFalse(link.exists())
                    yield {"status": "passed"}, FakeProcess(), original

                worker = threading.Thread(target=serve_request, daemon=True)
                worker.start()
                with patch.object(fit, "measure", measured), redirect_stdout(io.StringIO()):
                    self.assertEqual(fit.main(["--model", "x", "--serve", "--socket", str(link),
                                               "--report", str(report)]), 130)
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertFalse(link.is_symlink())
