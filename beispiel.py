"""Ein synthetischer Fehlerfall. Auf stdout landen ausschließlich geprüfte SFT-Zeilen."""

import argparse
from http.client import HTTPException
import json
import sys

from b300_feedback import generate
from glm_feedback import FeedbackCase, Verdict, prepare_sft


def verify_order(case_id, answer):
    # Der Sollwert kommt aus dem Test-Fixture zur Aufgabe.
    if case_id != "demo-001":
        return Verdict.UNKNOWN
    try:
        fields = json.loads(answer, object_pairs_hook=tuple)
    except (ValueError, RecursionError):
        return Verdict.FAIL
    # Tupel erhält doppelte Schlüssel und unterscheidet Objekte von Arrays.
    return Verdict.PASS if fields == (("order_id", "AB-17"),) else Verdict.FAIL


def local_glm(case, attempt):
    return generate(8000, case, attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm", action="store_true", help="Korrektur vom lokalen GLM anfordern")
    args = parser.parse_args()
    case = FeedbackCase(
        case_id="demo-001", group_id="demo-orders-a", split="train", training_allowed=True,
        prompt="Gib die Bestellnummer AB-17 als JSON-Objekt mit genau dem Feld order_id aus.",
        rejected='{"id":"AB-17"}', feedback_note="Der Feldname stimmt nicht.",
        correction=None if args.glm else '{"order_id":"AB-17"}',
    )
    try:
        decision = prepare_sft(case, verify_order, local_glm if args.glm else None)
    except (OSError, HTTPException, KeyError, IndexError, ValueError, TypeError) as error:
        parser.exit(2, f"GLM-Anfrage fehlgeschlagen ({type(error).__name__}).\n")
    print(decision.reason, file=sys.stderr)
    if decision.row is None:
        return 1
    print(json.dumps(decision.row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
