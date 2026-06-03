from certfe.dsl.lineage import LineageGraph

from .base import CertResult
from .policy import CertificatePolicy
from .schema_validity import verify_c1
from .no_leakage import verify_c2
from .temporal_validity import verify_c3
from .type_unit_safety import verify_c4
from .train_only_fitting import verify_c5
from .lineage_repr import verify_c6


def verify(graph: LineageGraph, policy: CertificatePolicy) -> CertResult:
    """Run all active certificates and return aggregate result."""
    traces = []

    if policy.c1_schema_validity:
        traces.append(verify_c1(graph, policy))
    if policy.c2_no_leakage:
        traces.append(verify_c2(graph, policy))
    if policy.c3_temporal_validity:
        traces.append(verify_c3(graph, policy))
    if policy.c4_type_unit_safety:
        traces.append(verify_c4(graph, policy))
    if policy.c5_train_only_fitting:
        traces.append(verify_c5(graph, policy))
    if policy.c6_lineage:
        traces.append(verify_c6(graph, policy))

    return CertResult(traces=traces)
