from certfe.dsl.lineage import LineageGraph
from certfe.dsl.operators import FIT_OPS

from .base import CertTrace, Verdict
from .policy import CertificatePolicy


def verify_c5(graph: LineageGraph, policy: CertificatePolicy) -> CertTrace:
    """C5: Train-Only Fitting — fit-ops must declare fit_split=train.

    On PromptFE IID datasets, this is the certificate that catches the default
    target-encoding leakage (fit on train+val before split). See Q1 protocol.
    """
    issues: list[str] = []

    for nid, node in graph.nodes.items():
        if node.op not in FIT_OPS:
            continue
        # Fit-ops with no explicit source_split inherit the split of their inputs.
        # If any ancestor came from val/test, that's a C5 fail.
        ancestors = graph.ancestors(nid)
        for aid in ancestors:
            anc = graph.nodes.get(aid)
            if anc is None:
                continue
            if anc.source_split and anc.source_split != policy.train_split_label:
                issues.append(
                    f"{nid} ({node.op.value}): fit-op has ancestor {aid} "
                    f"with source_split={anc.source_split}, must be "
                    f"'{policy.train_split_label}' only"
                )

    if issues:
        return CertTrace(
            cert_name="C5_TrainOnlyFitting",
            verdict=Verdict.FAIL,
            detail="; ".join(issues),
            witness=issues,
            failure_code="C5.FIT_LEAK",
        )
    return CertTrace(cert_name="C5_TrainOnlyFitting", verdict=Verdict.PASS)
