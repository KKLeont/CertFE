from certfe.dsl.lineage import LineageGraph
from certfe.dsl.types import DataType, NUMERIC_ONLY_OPS, CATEGORICAL_ONLY_OPS

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c4(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C4: Type/Unit Safety — type/unit derivation self-consistent.

    Checks:
    - Numeric-only ops receive numeric children
    - Categorical-only ops receive categorical children
    - Same-unit for Add/Subtract
    """
    issues: list[str] = []

    for nid, node in graph.nodes.items():
        for child_id in node.children:
            child = graph.nodes.get(child_id)
            if child is None:
                continue

            if child.dtype is None:
                continue

            op_name = node.op.value

            # Numeric-only check
            if op_name in NUMERIC_ONLY_OPS and child.dtype != DataType.NUMERIC:
                issues.append(
                    f"{nid} ({op_name}): child {child_id} has type "
                    f"{child.dtype.value}, expected numeric"
                )

            # Categorical-only check
            if op_name in CATEGORICAL_ONLY_OPS:
                if child.dtype != DataType.CATEGORICAL and child.col_index is not None:
                    issues.append(
                        f"{nid} ({op_name}): child {child_id} has type "
                        f"{child.dtype.value}, expected categorical"
                    )

            # Same-unit for Add/Subtract
            if op_name in ("Add", "Subtract"):
                siblings = [graph.nodes[c] for c in node.children if c != child_id and graph.nodes.get(c)]
                for sib in siblings:
                    if sib.unit and child.unit and sib.unit != child.unit:
                        issues.append(
                            f"{nid} ({op_name}): unit mismatch — "
                            f"{child.unit.value} vs {sib.unit.value}"
                        )

    if issues:
        return CertTrace(
            cert_name="C4_TypeUnitSafety",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
            failure_code="C4.TYPE",
        )
    return CertTrace(cert_name="C4_TypeUnitSafety", verdict=Verdict.PASS)
