"""Prepare verified JSON-answer SFT rows with independent local GLM replica workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
from http.client import HTTPConnection, HTTPException
import json
from pathlib import Path
import tempfile
import unicodedata

from glm_feedback import FeedbackCase, Verdict, prepare_sft


def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def _invalid_constant(_):
    raise ValueError("Non-finite JSON number")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant,
                      parse_int=Decimal, parse_float=Decimal)


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(v, right[k]) for k, v in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _objects(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = _json(line)
                if not isinstance(obj, dict):
                    raise ValueError("Expected an object")
            except (ValueError, RecursionError, ArithmeticError) as exc:
                raise ValueError(f"Invalid JSON object in {Path(path).name}, line {number}") from exc
            yield number, obj


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_cases(path):
    cases, identifiers, groups, prompts = [], set(), {}, {}
    for line, raw in _objects(path):
        try:
            case = FeedbackCase(**raw)
        except TypeError as exc:
            raise ValueError(f"Invalid feedback fields at line {line}") from exc
        for field in ("case_id", "group_id", "split", "prompt", "rejected"):
            if not isinstance(getattr(case, field), str):
                raise ValueError(f"Expected text in {field} at line {line}")
        for field in ("correction", "feedback_note"):
            if getattr(case, field) is not None and not isinstance(getattr(case, field), str):
                raise ValueError(f"Expected optional text in {field} at line {line}")
        if not all(v.strip() for v in (case.case_id, case.group_id, case.split)):
            raise ValueError(f"Empty identifier or split at line {line}")
        if type(case.training_allowed) is not bool or case.case_id in identifiers:
            raise ValueError(f"Invalid permission or duplicate case ID at line {line}")
        identifiers.add(case.case_id)
        for key, partitions in ((_normal(case.group_id), groups), (_normal(case.prompt), prompts)):
            if key and partitions.setdefault(key, case.split) != case.split:
                raise ValueError(f"Group or prompt occurs in different partitions at line {line}")
        cases.append(case)
    return cases


def read_oracles(path):
    result = {}
    for line, obj in _objects(path):
        if (set(obj) != {"case_id", "expected_json"} or not isinstance(obj["case_id"], str)
                or not obj["case_id"].strip() or not isinstance(obj["expected_json"], dict)
                or obj["case_id"] in result):
            raise ValueError(f"Invalid or duplicate JSON oracle at line {line}")
        result[obj["case_id"]] = obj["expected_json"]
    return result


def generate(port, case, attempt):
    evidence = {"task": case.prompt, "rejected": case.rejected, "feedback": case.feedback_note}
    body = {
        "model": "glm-5.3-flash", "max_tokens": 2048, "seed": attempt,
        "messages": [
            {"role": "system", "content": (
                "Solve the original task in the task field. Review the rejected answer and "
                "user feedback as evidence. Return only the corrected final answer."
            )},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ],
    }
    connection = HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"Replica on port {port} returned HTTP {response.status}")
        choice = json.load(response)["choices"][0]
        if choice["finish_reason"] != "stop":
            return None
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"Invalid final answer from replica on port {port}")
        return content
    finally:
        connection.close()


def prepare_batch(cases_file, oracles_file, out, ports=(8000,), *,
                  generator=generate):
    import fcntl  # Linux worker lock; the standalone inference client also works elsewhere.

    ports = tuple(ports)
    if (not ports or any(type(p) is not int or not 1 <= p <= 65535 for p in ports)
            or len(set(ports)) != len(ports)):
        raise ValueError("Supply distinct, valid loopback TCP ports")
    cases = read_cases(cases_file)
    oracles = read_oracles(oracles_file)

    def verify(identifier, answer):
        if identifier not in oracles:
            return Verdict.UNKNOWN
        try:
            valid = _same_json(_json(answer), oracles[identifier])
        except (ValueError, RecursionError, ArithmeticError):
            valid = False
        return Verdict.PASS if valid else Verdict.FAIL

    def worker(lane):
        indexed = []
        for i in range(lane, len(cases), len(ports)):
            decision = prepare_sft(cases[i], verify,
                                   lambda case, attempt: generator(ports[lane], case, attempt))
            indexed.append((i, decision))
        return indexed

    target = Path(out).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ("." + target.name + ".feedback.lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists() or target.is_symlink():
            raise ValueError("Choose a new dataset output directory")
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".feedback-") as tmp:
            ready = Path(tmp) / "dataset"
            ready.mkdir(mode=0o700)
            with ThreadPoolExecutor(max_workers=len(ports)) as pool:
                decisions = sorted(item for lane in pool.map(worker, range(len(ports))) for item in lane)
            accepted = 0
            with (ready / "train.jsonl").open("w", encoding="utf-8") as train, (
                    ready / "audit.jsonl").open("w", encoding="utf-8") as audit:
                for _, decision in decisions:
                    if decision.row is not None:
                        train.write(json.dumps(decision.row, ensure_ascii=False) + "\n")
                        accepted += 1
                    record = {
                        "case_id_sha256": hashlib.sha256(decision.case_id.encode()).hexdigest(),
                        "group_id_sha256": hashlib.sha256(decision.group_id.encode()).hexdigest(),
                        "reason": decision.reason, "source": decision.source,
                    }
                    audit.write(json.dumps(record) + "\n")
            ready.rename(target)
    return {"cases": len(cases), "accepted": accepted, "out": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--oracles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ports", nargs="+", type=int, default=[8000],
                        help="active local gateway port (default 8000), or explicit serving replicas")
    args = parser.parse_args()
    try:
        result = prepare_batch(args.cases, args.oracles, args.out, args.ports)
    except (OSError, HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
