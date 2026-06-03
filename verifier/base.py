from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "n/a"


class FailureCode(str, Enum):
    C1_UNKNOWN_COLUMN = "C1.UNKNOWN_COLUMN"
    C1_ARITY = "C1.ARITY"
    C2_TARGET_LEAK = "C2.TARGET_LEAK"
    C2_FORBIDDEN_LEAK = "C2.FORBIDDEN_LEAK"
    C3_FUTURE_TIME = "C3.FUTURE_TIME"
    C4_UNIT = "C4.UNIT"
    C4_TYPE = "C4.TYPE"
    C5_FIT_LEAK = "C5.FIT_LEAK"
    C6_NONDETERMINISTIC = "C6.NONDETERMINISTIC"
    C6_IO = "C6.IO"
    INTERNAL = "INTERNAL"


@dataclass
class CertTrace:
    """Single certificate's verdict with witness paths."""
    cert_name: str
    verdict: Verdict
    detail: str = ""
    witness: list[str] = field(default_factory=list)
    failure_code: str | None = None


@dataclass
class CertResult:
    """Aggregate result of all active certificates."""
    traces: list[CertTrace] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff all applicable certs are pass (n/a counts as pass)."""
        for t in self.traces:
            if t.verdict == Verdict.FAIL:
                return False
        return True

    @property
    def failures(self) -> list[CertTrace]:
        return [t for t in self.traces if t.verdict == Verdict.FAIL]

    @property
    def first_failure_code(self) -> str | None:
        for t in self.traces:
            if t.verdict == Verdict.FAIL and t.failure_code:
                return t.failure_code
        return None
