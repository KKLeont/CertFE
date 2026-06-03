"""CertFE runner — minimal agent loop."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from certfe.dsl.compile import compile_program, CompileError
from certfe.dsl.execute import evaluate as dsl_evaluate, EvaluationContext
from certfe.dsl.types import DataType
from certfe.evaluator import load_dataset, Evaluator, ProgressiveEvaluator
from certfe.generator.proposer import Proposer
from certfe.generator.coverage import CoverageMap
from certfe.verifier import verify
from certfe.verifier.policy import CertificatePolicy
from certfe.critic import critique
from certfe.redundancy import compute_redundancy, canonical_hash

logger = logging.getLogger(__name__)


@dataclass
class AcceptedFeature:
    feature_id: str
    family: str
    program_json: dict
    program_hash: str
    delta_metric: float
    col_train: pd.Series = field(repr=False)
    col_val: pd.Series = field(repr=False)
    declared_intent: str = ""
    interpretation: str = ""
    plausibility: str = ""
    intent_alignment: str = ""


@dataclass
class CertCheckTrace:
    name: str
    status: str  # pass / fail / n/a
    witness: list[str] = field(default_factory=list)
    detail: str = ""
    failure_code: str | None = None


@dataclass
class FeatureRecord:
    feature_id: str
    family: str
    program_json: dict
    status: str  # accepted | rejected_cert | rejected_eval
    rejection_reason: str = ""
    round_num: int = 0
    # Metadata
    program_hash: str = ""
    schema_hash: str = ""
    policy_version: str = "v1.0"
    prompt_version: str = "v1.0"
    # Hard layer
    hard_overall: str = ""  # pass / fail
    hard_traces: list[CertCheckTrace] = field(default_factory=list)
    # Soft.empirical
    delta_metric_mean: float = 0.0
    delta_metric_std: float = 0.0
    redundancy_max: float = 0.0
    redundancy_argmax: str = ""
    cost_gen_tokens: int = 0
    cost_verify_ms: float = 0.0
    cost_eval_seconds: float = 0.0
    # Soft.semantic
    plausibility: str = ""
    interpretation: str = ""
    declared_intent: str = ""
    intent_alignment: str = ""
    semantic_witness: str = ""
    source: str = ""

    @property
    def delta_metric(self) -> float:
        """Backward compat: stop_criterion code reads .delta_metric."""
        return self.delta_metric_mean

    @delta_metric.setter
    def delta_metric(self, v: float):
        self.delta_metric_mean = v


@dataclass
class RunState:
    dataset_name: str
    S: list[AcceptedFeature] = field(default_factory=list)
    coverage_map: CoverageMap = field(default_factory=CoverageMap)
    all_records: list[FeatureRecord] = field(default_factory=list)
    round_num: int = 0


def build_policy(bundle) -> CertificatePolicy:
    target_col = bundle.n_original_cols
    return CertificatePolicy(
        c1_schema_validity=True,
        c2_no_leakage=True,
        c3_temporal_validity=True,  # always run; will return n/a when prediction_time=None
        c4_type_unit_safety=True,
        c5_train_only_fitting=True,
        c6_lineage=True,
        column_types={i: bundle.col_types.get(i, "num") for i in range(bundle.n_original_cols)},
        column_units={},
        target_col=target_col,
        forbidden_cols=set(),
        prediction_time=None,
        split_col=None,
        train_split_label="train",
    )


def _extract_col_indices(prog_json: dict) -> frozenset[int]:
    """Walk a program JSON tree and collect all Col node indices."""
    indices = set()
    def walk(node):
        if isinstance(node, dict):
            if node.get("op") == "Col" and "col_index" in node:
                indices.add(int(node["col_index"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(prog_json)
    return frozenset(indices)


def execute_program_on_data(program, df: pd.DataFrame):
    """Execute a compiled program on a dataframe, return feature column."""
    try:
        ctx = EvaluationContext(df=df)
        col = dsl_evaluate(program, ctx)
        return col
    except Exception:
        logger.warning("Execution failed for %s", program.feature_id)
        return None


def _compute_schema_hash(bundle, policy: CertificatePolicy) -> str:
    """Hash of (schema, policy) for cert cache key."""
    import hashlib
    payload = {
        "dataset": bundle.name,
        "n_cols": bundle.n_original_cols,
        "col_types": bundle.col_types,
        "target_col": policy.target_col,
        "forbidden_cols": sorted(policy.forbidden_cols),
        "checks": [policy.c1_schema_validity, policy.c2_no_leakage, policy.c3_temporal_validity,
                   policy.c4_type_unit_safety, policy.c5_train_only_fitting, policy.c6_lineage],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _traces_from_cert(cert_result) -> list:
    """Convert verifier CertResult traces → CertCheckTrace list."""
    out = []
    for t in cert_result.traces:
        fc = getattr(t, "failure_code", None)
        if hasattr(fc, "value"):
            fc = fc.value
        out.append(CertCheckTrace(
            name=t.cert_name,
            status=t.verdict.value if hasattr(t.verdict, "value") else str(t.verdict),
            witness=list(t.witness) if t.witness else [],
            detail=t.detail or "",
            failure_code=fc,
        ))
    return out


def run(
    dataset_name: str,
    R_max: int = 50,
    J: int = 5,
    K_max: int = 30,
    epsilon_rel: float = 0.01,
    redundancy_threshold: float = 0.92,
    output_dir: str | None = None,
    disable_verifier: bool = False,
    disable_c5: bool = False,
    disable_coverage: bool = False,
    disable_progressive: bool = False,
    fixed_budget: bool = False,
    model_name: str = "RF",
    seed: int = 0,
    batch_size: int = 5,
    use_cv_baseline: bool = False,
) -> RunState:
    """Main CertFE agent loop."""
    bundle = load_dataset(dataset_name, seed=seed)
    evaluator = Evaluator(bundle, model_name=model_name, use_cv_baseline=use_cv_baseline)
    progressive = ProgressiveEvaluator(evaluator, epsilon_rel=epsilon_rel)
    policy = build_policy(bundle)

    if disable_verifier:
        policy.c1_schema_validity = False
        policy.c2_no_leakage = False
        policy.c3_temporal_validity = False
        policy.c4_type_unit_safety = False
        policy.c5_train_only_fitting = False
        policy.c6_lineage = False
        logger.info("ABLATION: verifier disabled (all C1-C6 off)")

    if disable_c5:
        policy.c5_train_only_fitting = False
        logger.info("ABLATION: C5 (train-only fitting) disabled")

    if disable_coverage:
        logger.info("ABLATION: coverage map disabled")

    if disable_progressive:
        logger.info("ABLATION: progressive evaluation disabled (single-tier)")
        epsilon_rel = -float("inf")  # accept everything through Tier 3

    if fixed_budget:
        logger.info("ABLATION: fixed budget mode (no J-based stopping)")

    col_info = {}
    for i in range(bundle.n_original_cols):
        col_info[i] = {
            "type": bundle.col_types.get(i, "num"),
            "desc": bundle.col_descriptions[i] if bundle.col_descriptions and i < len(bundle.col_descriptions) else "",
        }

    _type_map = {"num": DataType.NUMERIC, "cat": DataType.CATEGORICAL}
    col_types_map = {i: _type_map.get(bundle.col_types.get(i, "num"), DataType.NUMERIC) for i in range(bundle.n_original_cols)}
    col_types_map[bundle.n_original_cols] = DataType.NUMERIC  # target col — compile won't crash, C2 will reject

    proposer = Proposer(
        dataset_name=dataset_name,
        column_info=col_info,
        target_col=bundle.n_original_cols,
        task="C" if bundle.task == "classification" else "R",
    )

    state = RunState(dataset_name=dataset_name)
    rounds_without_accept = 0

    schema_hash = _compute_schema_hash(bundle, policy)
    POLICY_VERSION = "v1.0"
    PROMPT_VERSION = "v1.0"

    logger.info("Starting CertFE run on %s (task=%s, cols=%d)", dataset_name, bundle.task, bundle.n_original_cols)
    logger.info("Baseline score: %.4f", evaluator.baseline_score)

    for r in range(R_max):
        state.round_num = r
        accepted_this_round = False

        cmap = CoverageMap() if disable_coverage else state.coverage_map
        proposal = proposer.propose(
            coverage_map=cmap,
            accepted_features=[f.feature_id for f in state.S],
            batch_size=batch_size,
        )
        programs = proposal.programs
        per_program_tokens = proposal.token_usage // max(len(programs), 1) if programs else 0

        for prog in programs:
            record = FeatureRecord(
                feature_id=prog.feature_id,
                family=prog.family.value if hasattr(prog.family, 'value') else str(prog.family),
                program_json={},
                status="pending",
                round_num=r,
                schema_hash=schema_hash,
                policy_version=POLICY_VERSION,
                prompt_version=PROMPT_VERSION,
                cost_gen_tokens=per_program_tokens,
                declared_intent=proposal.declared_intents.get(prog.feature_id, ""),
                source="llm_critic_v1",
            )

            t_ver = time.time()
            try:
                ast, graph = compile_program(prog, col_types_map, target_col=bundle.n_original_cols)
                col_group = ast.col_indices if hasattr(ast, 'col_indices') else frozenset()
                cert_result = verify(graph, policy)
            except CompileError as e:
                record.status = "rejected_cert"
                err_msg = str(e)
                if "not in schema" in err_msg:
                    record.rejection_reason = "C1.UNKNOWN_COLUMN"
                elif "type" in err_msg.lower() or "numeric" in err_msg.lower() or "categorical" in err_msg.lower():
                    record.rejection_reason = "C4.TYPE"
                elif "unit" in err_msg.lower():
                    record.rejection_reason = "C4.UNIT"
                else:
                    record.rejection_reason = f"C1.COMPILE_ERROR"
                record.cost_verify_ms = (time.time() - t_ver) * 1000
                state.all_records.append(record)
                state.coverage_map.record_rejected(prog.family, frozenset())
                proposer.add_rejection(prog.feature_id, f"{record.rejection_reason}: {err_msg[:80]}")
                continue
            except Exception as e:
                record.status = "rejected_cert"
                record.rejection_reason = "INTERNAL"
                record.cost_verify_ms = (time.time() - t_ver) * 1000
                state.all_records.append(record)
                state.coverage_map.record_rejected(prog.family, frozenset())
                proposer.add_rejection(prog.feature_id, f"INTERNAL: {str(e)[:80]}")
                continue

            record.cost_verify_ms = (time.time() - t_ver) * 1000
            record.hard_traces = _traces_from_cert(cert_result)
            record.hard_overall = "pass" if cert_result.passed else "fail"

            if not cert_result.passed:
                record.status = "rejected_cert"
                record.rejection_reason = cert_result.first_failure_code or "UNKNOWN"
                state.all_records.append(record)
                state.coverage_map.record_rejected(prog.family, col_group)
                proposer.add_rejection(prog.feature_id, record.rejection_reason)
                continue

            col_train = execute_program_on_data(prog, bundle.x_train)
            col_val = execute_program_on_data(prog, bundle.x_val)

            if col_train is None or col_val is None:
                record.status = "rejected_eval"
                record.rejection_reason = "EXECUTION_FAILED"
                state.all_records.append(record)
                continue

            # --- Critic (semantic layer) ---
            declared_intent = proposal.declared_intents.get(prog.feature_id, "")
            prog_json_for_critic = {"op": prog.root.__class__.__name__, "feature_id": prog.feature_id}
            try:
                from certfe.dsl.grammar import program_to_json
                prog_json_for_critic = program_to_json(prog).get("program", {})
            except Exception:
                pass

            record.program_json = prog_json_for_critic

            semantic = None
            try:
                semantic = critique(
                    program_json=prog_json_for_critic,
                    column_descriptions={i: col_info[i].get("desc", "") for i in col_info},
                    declared_intent=declared_intent,
                )
            except Exception as e:
                logger.debug("Critic failed for %s: %s", prog.feature_id, e)

            if semantic:
                record.plausibility = semantic.plausibility
                record.interpretation = semantic.interpretation
                record.intent_alignment = semantic.intent_alignment
                record.semantic_witness = semantic.plausibility_witness

            # --- Redundancy check ---
            p_hash = canonical_hash(prog_json_for_critic)
            record.program_hash = p_hash
            accepted_for_red = [(f.feature_id, f.col_train, f.program_hash) for f in state.S]
            red_result = compute_redundancy(col_train, p_hash, accepted_for_red)
            record.redundancy_max = red_result.redundancy_max
            record.redundancy_argmax = red_result.redundancy_argmax

            if red_result.redundancy_max >= redundancy_threshold:
                record.status = "rejected_eval"
                record.rejection_reason = "POST_EVAL_REDUNDANT"
                record.delta_metric = 0.0
                state.all_records.append(record)
                state.coverage_map.record_rejected(prog.family, col_group)
                continue

            # --- Progressive evaluation ---
            existing_train = [f.col_train for f in state.S]
            existing_val = [f.col_val for f in state.S]

            eval_result = progressive.evaluate(
                col_train, col_val, existing_train, existing_val,
            )
            record.cost_eval_seconds = eval_result.cost_seconds
            record.delta_metric_mean = eval_result.delta_metric
            record.delta_metric_std = eval_result.delta_metric_std

            if eval_result.passed:
                record.status = "accepted"
                accepted_feature = AcceptedFeature(
                    feature_id=prog.feature_id,
                    family=prog.family.value if hasattr(prog.family, 'value') else str(prog.family),
                    program_json=prog_json_for_critic,
                    program_hash=p_hash,
                    delta_metric=eval_result.delta_metric,
                    col_train=col_train,
                    col_val=col_val,
                    declared_intent=declared_intent,
                    interpretation=semantic.interpretation if semantic else "",
                    plausibility=semantic.plausibility if semantic else "",
                    intent_alignment=semantic.intent_alignment if semantic else "",
                )
                state.S.append(accepted_feature)
                state.coverage_map.record_accepted(prog.family, col_group)
                accepted_this_round = True
                logger.info("Round %d: ACCEPTED %s (Δ=%.4f, |S|=%d cols=%s)", r, prog.feature_id, eval_result.delta_metric, len(state.S), set(col_group))
            else:
                record.status = "rejected_eval"
                record.rejection_reason = eval_result.rejection_reason
                state.coverage_map.record_rejected(prog.family, col_group)

            state.all_records.append(record)

        if accepted_this_round:
            rounds_without_accept = 0
        else:
            rounds_without_accept += 1

        if not fixed_budget:
            if rounds_without_accept >= J:
                logger.info("Stopping: %d consecutive rounds without accept", J)
                break
            if len(state.S) >= K_max:
                logger.info("Stopping: reached K_max=%d features", K_max)
                break

    if fixed_budget:
        stop_reason = "fixed_budget"
    else:
        stop_reason = "budget" if state.round_num >= R_max - 1 else (
            "no_accept" if rounds_without_accept >= J else "feature_cap"
        )

    if output_dir:
        final_score = _compute_final_score(state, evaluator, bundle)
        _write_output(state, evaluator, stop_reason, output_dir, final_score)

    logger.info("Run complete: |S|=%d, rounds=%d, stop=%s", len(state.S), state.round_num + 1, stop_reason)
    if state.S:
        final_score_val = _compute_final_score(state, evaluator, bundle)
        logger.info("FINAL RESULT: baseline=%.4f → with CertFE features=%.4f (Δ=%.4f)",
                    evaluator.baseline_score, final_score_val,
                    final_score_val - evaluator.baseline_score)
    return state


def _compute_final_score(state: RunState, evaluator: Evaluator, bundle) -> float:
    """Evaluate all accepted features together on test set."""
    if not state.S:
        return evaluator.baseline_score

    x_train_full = pd.concat([bundle.x_train, bundle.x_val]).copy()
    x_test = bundle.x_test.copy()
    y_train_full = pd.concat([bundle.y_train, bundle.y_val])
    y_test = bundle.y_test

    for i, feat in enumerate(state.S):
        col_full = pd.concat([feat.col_train, feat.col_val])
        x_train_full[f"certfe_{i}"] = col_full.reindex(x_train_full.index)
        col_test = execute_program_on_data(
            _find_prog_placeholder(feat), bundle.x_test
        )
        if col_test is not None:
            x_test[f"certfe_{i}"] = col_test.reindex(x_test.index)
        else:
            x_test[f"certfe_{i}"] = 0.0

    return evaluator._score(x_train_full, x_test, y_train=y_train_full, y_eval=y_test)


def _find_prog_placeholder(feat: AcceptedFeature):
    """Reconstruct a minimal program object for re-execution on test."""
    from certfe.dsl.grammar import program_from_json
    try:
        prog_json = {
            "feature_id": feat.feature_id,
            "family": feat.family,
            "program": feat.program_json,
            "output": {"name": feat.feature_id, "type": "numeric", "unit": "unitless"},
        }
        return program_from_json(prog_json)
    except Exception:
        return None


def _write_output(state: RunState, evaluator: Evaluator, stop_reason: str, output_dir: str, final_score: float = None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "features.jsonl", "w") as f:
        for rec in state.all_records:
            cert = {
                "feature_id": rec.feature_id,
                "family": rec.family,
                "round": rec.round_num,
                "status": rec.status,
                "rejection_reason": rec.rejection_reason,
                "program": rec.program_json,
                "prompt_version": rec.prompt_version,
                "hard": {
                    "program_hash": rec.program_hash,
                    "schema_hash": rec.schema_hash,
                    "policy_version": rec.policy_version,
                    "overall": rec.hard_overall,
                    "checks": [
                        {
                            "name": t.name,
                            "status": t.status,
                            "witness": t.witness,
                            "detail": t.detail,
                            "failure_code": t.failure_code,
                        } for t in rec.hard_traces
                    ],
                },
                "soft": {
                    "empirical": {
                        "delta_metric_mean": rec.delta_metric_mean,
                        "delta_metric_std": rec.delta_metric_std,
                        "redundancy_max": rec.redundancy_max,
                        "redundancy_argmax": rec.redundancy_argmax,
                        "cost_gen_tokens": rec.cost_gen_tokens,
                        "cost_verify_ms": rec.cost_verify_ms,
                        "cost_eval_seconds": rec.cost_eval_seconds,
                    },
                    "semantic": {
                        "plausibility": rec.plausibility,
                        "interpretation": rec.interpretation,
                        "declared_intent": rec.declared_intent,
                        "intent_alignment": rec.intent_alignment,
                        "witness": rec.semantic_witness,
                        "source": rec.source,
                    } if rec.status != "rejected_cert" else None,
                },
            }
            f.write(json.dumps(cert) + "\n")

    cpuf_data = []
    cumulative_cost = 0.0
    k = 0
    for rec in state.all_records:
        cumulative_cost += rec.cost_verify_ms / 1000 + rec.cost_eval_seconds
        if rec.status == "accepted":
            k += 1
            cpuf_data.append({"k": k, "cumulative_cost_seconds": cumulative_cost})

    with open(out / "cpuf.json", "w") as f:
        json.dump(cpuf_data, f, indent=2)

    metrics = {
        "dataset": state.dataset_name,
        "S_size": len(state.S),
        "total_rounds": state.round_num + 1,
        "stop_reason": stop_reason,
        "baseline_score": evaluator.baseline_score,
        "final_score_test": final_score,
        "improvement": (final_score - evaluator.baseline_score) if final_score else 0.0,
        "total_candidates": len(state.all_records),
        "accepted": sum(1 for r in state.all_records if r.status == "accepted"),
        "rejected_cert": sum(1 for r in state.all_records if r.status == "rejected_cert"),
        "rejected_eval": sum(1 for r in state.all_records if r.status == "rejected_eval"),
        "accepted_features": [f.feature_id for f in state.S],
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
