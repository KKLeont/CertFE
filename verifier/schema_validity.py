from certfe.dsl.lineage import LineageGraph
from certfe.dsl.operators import Operator, OPERATOR_REGISTRY

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c1(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C1: Schema Validity — all column refs exist; operators match signatures."""
    issues: list[str] = []

    for nid, node in graph.nodes.items():
        sig = OPERATOR_REGISTRY.get(node.op)
        if sig is None:
            issues.append(f"{nid}: unknown operator {node.op}")
            continue

        # Check column refs exist
        if node.col_index is not None and node.col_index not in policy.column_types:
            issues.append(
                f"{nid}: column {node.col_index} not in schema"
            )

        # Check arity
        expected_arity = sig.arity
        actual_arity = len(node.children)
        if expected_arity > 0 and actual_arity != expected_arity:
            issues.append(
                f"{nid}: {node.op.value} expects {expected_arity} children, got {actual_arity}"
            )

    if issues:
        return CertTrace(
            cert_name="C1_SchemaValidity",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
            failure_code="C1.SCHEMA",
        )
    return CertTrace(cert_name="C1_SchemaValidity", verdict=Verdict.PASS)
