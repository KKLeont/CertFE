"""Prompt templates for CertFE feature program generator."""
from __future__ import annotations

from certfe.dsl.family import IID_FAMILIES, FeatureFamily
from certfe.dsl.operators import OPERATOR_REGISTRY


def build_system_prompt(
    dataset_name: str,
    column_info: dict[int, dict],
    target_col: int,
    task: str,
    applicable_families: frozenset[FeatureFamily] | None = None,
) -> str:
    families = applicable_families or IID_FAMILIES

    col_desc_lines = []
    for ci in sorted(column_info.keys()):
        info = column_info[ci]
        role = ""
        if ci == target_col:
            role = " [TARGET — do NOT use except in TargetEncode.label_col]"
        col_desc_lines.append(f"  col_index={ci}: {info.get('type', 'num')} — {info.get('desc', '')}{role}")

    family_list = "\n".join(f"  - {f.value}" for f in sorted(families, key=lambda x: x.value))

    return f"""You are a feature engineering expert. Given a tabular dataset, you propose concrete feature programs in a restricted JSON DSL.

## Dataset: {dataset_name}
Task: {'classification' if task == 'C' else 'regression'}
Target column: col_index={target_col} (FORBIDDEN in feature programs except as TargetEncode label)

## Column schema
{chr(10).join(col_desc_lines)}

## Available feature families
{family_list}

## DSL reference — read carefully to avoid TYPE ERRORS

### Leaf operators
- Col: {{"op": "Col", "col_index": <integer>}} — MUST be an integer from the schema above
- Const: {{"op": "Const", "value": <number>, "dtype": "numeric"}}

### NUMERIC-ONLY operators — child/children MUST be numeric (check col type!)
- Unary: Log1p, Sqrt, Abs, Square, Reciprocal
  → {{"op": "...", "child": <numeric node>}}
- Scale/Fit (fit on train only): ZScore, MinMaxScale, QuantileBucketize
  → {{"op": "...", "child": <numeric Col>}}
- Binary arithmetic: Add, Subtract, Multiply, Divide, Ratio
  → {{"op": "...", "left": <numeric node>, "right": <numeric node>}}
- Row-wise: RowMean, RowMax, RowMin, RowSum, RowStd
  → {{"op": "...", "children": [<numeric nodes...>]}}

### CATEGORICAL-INPUT operators — child MUST be a categorical Col
- FrequencyEncode (fit on train only): {{"op": "FrequencyEncode", "child": {{"op": "Col", "col_index": <cat col>}}}}
- Cross: {{"op": "Cross", "left": {{"op": "Col", "col_index": <cat>}}, "right": {{"op": "Col", "col_index": <cat>}}}}
- IsEqual: {{"op": "IsEqual", "left": {{"op": "Col", "col_index": <any>}}, "right": {{"op": "Const", "value": <val>, "dtype": "numeric"}}}}
- TargetEncode: {{"op": "TargetEncode", "cat_col": {{"op": "Col", "col_index": <cat>}}, "label_col": {{"op": "Col", "col_index": {target_col}}}}}

### ANY-TYPE operators
- IsNull: {{"op": "IsNull", "child": <node>}} — works on any type
- Coalesce: {{"op": "Coalesce", "left": <node>, "right": {{"op": "Const", "value": <val>, "dtype": "numeric"}}}} — replaces null with const

## CRITICAL RULES
1. col_index MUST be an integer from the schema (0 to {max(column_info.keys())}).
2. NEVER reference col_index={target_col} except as label_col in TargetEncode.
3. Check column TYPES before choosing operators: numeric ops (Log1p,Sqrt,Add,Multiply,RowMean...) require NUMERIC children. Categorical ops (FrequencyEncode,Cross) require CATEGORICAL Col.
4. Each program MUST include a "declared_intent" field (≥10 characters) explaining what the feature captures.
5. The coverage map below shows per-cell status. Prioritize cells marked "NOT YET EXPLORED" or with low yield.
6. Return ONLY a JSON array. No markdown, no explanation.

## Output format (JSON array of objects)
[
  {{
    "feature_id": "F_<descriptive_name>",
    "family": "<one of the available families>",
    "declared_intent": "<what this feature captures, ≥10 chars>",
    "program": {{"op": "...", ...}},
    "output": {{"name": "<feature_name>", "type": "numeric", "unit": "unitless"}}
  }}
]
"""


def build_user_prompt(
    coverage_context: str,
    current_features: list[str] | None = None,
    recent_rejections: list[dict] | None = None,
    batch_size: int = 5,
) -> str:
    existing = "\n".join(f"  - {f}" for f in (current_features or [])) or "  (none yet)"

    rejection_lines = ""
    if recent_rejections:
        rej_summary = []
        for r in recent_rejections[:10]:
            rej_summary.append(f"  - {r.get('feature_id', '?')}: {r.get('reason', '?')}")
        rejection_lines = f"\n## Recent rejections (avoid these patterns)\n" + "\n".join(rej_summary)

    return f"""## Coverage map (full 25-cell status)
{coverage_context}

## Already accepted features
{existing}
{rejection_lines}

## Instruction
Propose {batch_size} new feature programs. Prioritize under-explored cells.
Each program MUST include "declared_intent" (≥10 characters).
Return ONLY a JSON array, no other text."""


def build_coverage_context(
    coverage_map,
    all_cols: set[int],
) -> str:
    """Coverage dump: per-family stats with column-group count."""
    from collections import defaultdict

    # Aggregate per family
    fam_stats: dict[str, dict] = defaultdict(lambda: {"n_acc": 0, "n_rej": 0, "col_groups": set()})
    for (fam_val, col_group), entry in coverage_map.entries.items():
        s = fam_stats[fam_val]
        s["n_acc"] += entry.n_accepted
        s["n_rej"] += entry.n_rejected
        if col_group:
            s["col_groups"].add(col_group)

    lines = []
    for fam in sorted(IID_FAMILIES, key=lambda x: x.value):
        stats = fam_stats.get(fam.value)
        if stats and (stats["n_acc"] + stats["n_rej"]) > 0:
            total = stats["n_acc"] + stats["n_rej"]
            rate = stats["n_acc"] / total if total > 0 else 0
            ng = len(stats["col_groups"])
            status = f"{stats['n_acc']}✓/{stats['n_rej']}✗, yield={rate:.0%}, {ng} col-groups"
            if rate == 0 and stats["n_rej"] > 0:
                status += " ← AVOID"
        else:
            status = "NOT YET EXPLORED ← PRIORITY"
        lines.append(f"  {fam.value}: {status}")

    return "\n".join(lines)
