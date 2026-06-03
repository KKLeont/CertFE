from certfe.dsl.lineage import LineageGraph

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c6(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C6: Lineage / Reproducibility — DAG is deterministic, contains no random nodes.

    In v1, all operators are deterministic so this is always pass unless the
    graph is cyclic or has dangling references.
    """
    # Check for cycles (should not happen with tree-built DAG but defensive)
    visited: set[str] = set()
    path: set[str] = set()
    issues: list[str] = []

    def _dfs(nid: str) -> bool:
        if nid in path:
            issues.append(f"cycle detected at {nid}")
            return False
        if nid in visited:
            return True
        path.add(nid)
        visited.add(nid)
        node = graph.nodes.get(nid)
        if node is not None:
            for child_id in node.children:
                if child_id not in graph.nodes:
                    issues.append(f"{nid}: dangling reference to {child_id}")
                else:
                    _dfs(child_id)
        path.discard(nid)
        return True

    if graph.root_id:
        _dfs(graph.root_id)

    if issues:
        return CertTrace(
            cert_name="C6_Lineage",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
            failure_code="C6.LINEAGE",
        )
    return CertTrace(cert_name="C6_Lineage", verdict=Verdict.PASS)
