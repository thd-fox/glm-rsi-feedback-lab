"""Compare two local GLM deployments on identical, held-out JSON extraction tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import re
import tempfile

from b300_feedback import _json, _normal, _same_json


def _rows(path, digest):
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                yield number, _json(line.decode("utf-8"))
            except (ValueError, RecursionError, ArithmeticError) as exc:
                raise ValueError(f"Invalid JSONL in {Path(path).name}, line {number}") from exc


def _fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inputs(checks, training_data):
    training_hash, checks_hash = hashlib.sha256(), hashlib.sha256()
    prompts, identifiers = set(), set()
    for line, row in _rows(training_data, training_hash):
        try:
            messages = row["messages"]
            if ([m["role"] for m in messages] != ["user", "assistant"]
                    or not all(isinstance(m["content"], str) and m["content"].strip() for m in messages)):
                raise ValueError
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid single-turn training row at line {line}") from exc
        prompts.add(_fingerprint(_normal(messages[0]["content"])))
    if not prompts:
        raise ValueError("The training dataset is empty")
    heldout = []
    for line, case in _rows(checks, checks_hash):
        if (not isinstance(case, dict)
                or set(case) != {"case_id", "group_id", "kind", "prompt", "expected_json"}
                or case["kind"] not in ("target", "control")
                or not isinstance(case["expected_json"], dict)
                or not all(isinstance(case[k], str) and case[k].strip()
                           for k in ("case_id", "group_id", "prompt"))):
            raise ValueError(f"Invalid holdout case at line {line}")
        key = _fingerprint(_normal(case["prompt"]))
        identifier = _normal(case["case_id"])
        if key in prompts or identifier in identifiers:
            raise ValueError(f"Duplicate holdout case or overlapping training prompt at line {line}")
        prompts.add(key)
        identifiers.add(identifier)
        heldout.append(case)
    if {case["kind"] for case in heldout} != {"target", "control"}:
        raise ValueError("Expected at least one target and one control in the holdout")
    return heldout, checks_hash.hexdigest(), training_hash.hexdigest()


def request_answer(port, body):
    conn = HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"Evaluation endpoint on port {port} returned HTTP {response.status}")
        value = json.load(response)
        if value.get("model") != "glm-5.3-flash":
            raise ValueError(f"Unexpected response model on port {port}")
        choice = value["choices"][0]
        if choice["finish_reason"] != "stop":
            return None
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError(f"Missing final answer on port {port}")
        return text
    finally:
        conn.close()


def _correct(text, expected):
    if text is None:
        return False
    try:
        return _same_json(_json(text), expected)
    except (ValueError, RecursionError, ArithmeticError):
        return False


def evaluate(base_port, candidate_port, base_sha256, candidate_sha256, checks, training_data,
             *, request=request_answer):
    ports = (base_port, candidate_port)
    if (base_port == candidate_port or any(type(p) is not int or not 1 <= p <= 65535 for p in ports)
            or base_sha256 == candidate_sha256
            or not all(isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h)
                       for h in (base_sha256, candidate_sha256))):
        raise ValueError("Evaluation requires two different ports and two distinct SHA-256 identities")
    cases, checks_digest, data_digest = load_inputs(checks, training_data)
    settings = {"model": "glm-5.3-flash", "temperature": 0, "top_p": 1, "max_tokens": 2048}
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for case in cases:
            case_hash = _fingerprint(case["case_id"])
            seed = int(case_hash[:8], 16) % (2**31)
            payload = json.dumps({**settings, "seed": seed,
                                  "messages": [{"role": "user", "content": case["prompt"]}]}).encode()
            futures = [pool.submit(request, port, payload) for port in ports]
            base, candidate = [_correct(future.result(), case["expected_json"]) for future in futures]
            results.append({"case_id_sha256": case_hash, "group_id_sha256": _fingerprint(case["group_id"]),
                            "kind": case["kind"], "seed": seed, "base": base, "candidate": candidate})
    fixes = sum(r["kind"] == "target" and not r["base"] and r["candidate"] for r in results)
    regressions = sum(r["base"] and not r["candidate"] for r in results)
    controls = sum(r["kind"] == "control" and r["base"] and r["candidate"] for r in results)
    passed = fixes > 0 and controls > 0 and regressions == 0
    return {
        "status": "passed" if passed else "rejected", "base_port": base_port,
        "candidate_port": candidate_port, "base_sha256": base_sha256,
        "candidate_sha256": candidate_sha256, "checks_sha256": checks_digest,
        "training_data_sha256": data_digest, "target_fixes": fixes, "regressions": regressions,
        "controls_passed": controls, "base_correct": sum(r["base"] for r in results),
        "candidate_correct": sum(r["candidate"] for r in results),
        "settings": settings, "cases": results,
    }


def evaluate_to_report(path, *args, **kwargs):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Evaluation report already exists")
    result = evaluate(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as temporary:
        json.dump(result, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--candidate-port", type=int, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--checks", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_to_report(args.report, args.base_port, args.candidate_port,
                                    args.base_sha256, args.candidate_sha256, args.checks,
                                    args.training_data)
    except (OSError, HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({k: result[k] for k in ("status", "target_fixes", "regressions", "controls_passed")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
