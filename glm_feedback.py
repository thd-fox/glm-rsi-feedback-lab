"""Build single-turn text SFT rows from sanitized, prepartitioned negative feedback.

Callers provide an independent case verifier and optionally a candidate generator.
The module does not contact a service. Callback exceptions propagate to the caller.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Literal, Optional, TypedDict


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeedbackCase:
    case_id: str
    group_id: str
    split: str
    training_allowed: bool
    prompt: str
    rejected: str
    correction: Optional[str] = None
    feedback_note: Optional[str] = None


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class SFTRow(TypedDict):
    messages: list[Message]


@dataclass(frozen=True)
class Decision:
    case_id: str
    group_id: str
    reason: str
    source: Optional[str] = None
    row: Optional[SFTRow] = None


def _has_text(value: Optional[str]) -> bool:
    return isinstance(value, str) and bool(value.strip())


def prepare_sft(
    case: FeedbackCase,
    verify: Callable[[str, str], Verdict],
    generate: Optional[Callable[[FeedbackCase, int], Optional[str]]] = None,
    *,
    max_attempts: int = 3,
) -> Decision:
    """Try the supplied correction first, then up to max_attempts generated replies.

    The verifier looks up an independent criterion by case_id. An ungradable
    original cannot yield a row. Generator attempts are numbered starting at one.
    The first passing candidate wins, preserving the original text exactly.
    """
    if type(max_attempts) is not int or max_attempts < 0:
        raise ValueError("max_attempts must be a nonnegative integer")

    def skip(reason: str) -> Decision:
        return Decision(case.case_id, case.group_id, reason)

    if case.training_allowed is not True:
        return skip("training_disallowed")
    if case.split != "train":
        return skip("outside_train")
    if not all(_has_text(s) for s in (case.case_id, case.group_id, case.prompt, case.rejected)):
        return skip("incomplete_case")

    original = verify(case.case_id, case.rejected)
    if original is Verdict.PASS:
        return skip("original_passed")
    if original is not Verdict.FAIL:
        return skip("original_unverified")

    def candidates() -> Iterator[tuple[Optional[str], str]]:
        yield case.correction, "provided_correction"
        if generate is not None:
            for attempt in range(1, max_attempts + 1):
                yield generate(case, attempt), "generated"

    seen = {case.rejected}
    unverified = False
    for text, source in candidates():
        if not isinstance(text, str) or not text.strip() or text in seen:
            continue
        seen.add(text)
        verdict = verify(case.case_id, text)
        if verdict is Verdict.PASS:
            row: SFTRow = {
                "messages": [
                    {"role": "user", "content": case.prompt},
                    {"role": "assistant", "content": text},
                ]
            }
            return Decision(case.case_id, case.group_id, "accepted", source, row)
        if verdict is not Verdict.FAIL:
            unverified = True

    if len(seen) == 1:
        return skip("no_alternative")
    return skip("candidate_unverified" if unverified else "candidates_failed")
