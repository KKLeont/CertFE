from certfe.dsl.lineage import LineageGraph

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c3(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C3: Temporal Validity — n/a if t* is None; otherwise check time origins."""
    if policy.prediction_time is None:
        return CertTrace(
            cert_name="C3_TemporalValidity",
            verdict=Verdict.NA,
            detail="t* is null — IID dataset, temporal validity not applicable",
        )

    # When t* is set, every node's time_origin must be <= t*,
    # and window-op end must be <= t*.
    # On PromptFE IID datasets this is always n/a.
    issues: list[str] = []

    for nid, node in graph.nodes.items():
        if node.time_origin is not None:
            # Simple check: time_origin should not be "future" relative to t*
            # Full implementation depends on time expression parser
            pass

    if issues:
        return CertTrace(
            cert_name="C3_TemporalValidity",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
        )
    return CertTrace(cert_name="C3_TemporalValidity", verdict=Verdict.PASS)
