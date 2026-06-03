from certfe.dsl.lineage import LineageGraph
from certfe.dsl.operators import Y_ALLOWED_OPS

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c2(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C2: No-Leakage — no target or forbidden columns in ancestors, except y-allowed slots."""
    prohibited: set[int] = {policy.target_col} if policy.target_col is not None else set()
    prohibited.update(policy.forbidden_cols)

    issues: list[str] = []

    for nid, node in graph.nodes.items():
        # Check the node itself (not just ancestors).
        # Skip if this Col node is a child of a Y-allowed op (TargetEncode label_col).
        if node.col_index is not None and node.col_index in prohibited:
            parent_is_y_allowed = False
            for pid, pnode in graph.nodes.items():
                if nid in pnode.children and pnode.op in Y_ALLOWED_OPS:
                    parent_is_y_allowed = True
                    break
            if not parent_is_y_allowed:
                issues.append(f"{nid} ({node.op.value}): directly references prohibited col {node.col_index}")

        is_y_allowed = node.op in Y_ALLOWED_OPS
        ancestors = graph.ancestors(nid)
        for ancestor_id in ancestors:
            anc = graph.nodes.get(ancestor_id)
            if anc is None or anc.col_index is None:
                continue
            if anc.col_index in prohibited:
                if is_y_allowed:
                    # TargetEncode label_col is y-allowed — check that this
                    # specific ancestor is the label slot, not hidden elsewhere
                    if node.op in Y_ALLOWED_OPS and ancestor_id in node.children:
                        continue  # explicit label slot — allowed
                issues.append(
                    f"{nid} ({node.op.value}): ancestor {ancestor_id} "
                    f"references prohibited col {anc.col_index}"
                )

    if issues:
        return CertTrace(
            cert_name="C2_NoLeakage",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
            failure_code="C2.TARGET_LEAK",
        )
    return CertTrace(cert_name="C2_NoLeakage", verdict=Verdict.PASS)
