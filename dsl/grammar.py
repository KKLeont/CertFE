from __future__ import annotations

import json
from typing import Any

from .operators import Operator, OPERATOR_REGISTRY, get_signature
from .family import FeatureFamily
from .types import DataType, Unit
from .ast import (
    ASTNode, ColRef, ConstVal, UnaryOp, BinaryOp, VariadicOp,
    TargetEncodeNode, FeatureProgram,
)


class GrammarError(Exception):
    """JSON does not conform to DSL grammar."""


def program_from_json(data: dict | str) -> FeatureProgram:
    """Parse a JSON DSL program into a FeatureProgram AST."""
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise GrammarError("top-level must be a JSON object")

    feature_id = data.get("feature_id", "")
    if not feature_id:
        raise GrammarError("missing feature_id")

    family_raw = data.get("family", "")
    try:
        family = FeatureFamily(family_raw)
    except ValueError:
        raise GrammarError(f"unknown family: {family_raw}")

    program_json = data.get("program")
    if program_json is None:
        raise GrammarError("missing program")
    root = _parse_node(program_json)

    output = data.get("output", {})
    output_name = output.get("name", feature_id)
    try:
        output_dtype = DataType(output.get("type", "numeric")) if output.get("type") else None
    except ValueError:
        output_dtype = DataType.NUMERIC
    try:
        output_unit = Unit(output.get("unit", "unitless")) if output.get("unit") else None
    except ValueError:
        output_unit = Unit.UNITLESS

    return FeatureProgram(
        feature_id=feature_id,
        family=family,
        root=root,
        output_name=output_name,
        output_dtype=output_dtype,
        output_unit=output_unit,
    )


def program_to_json(prog: FeatureProgram) -> dict:
    """Serialize a FeatureProgram to a JSON-serializable dict."""
    return {
        "feature_id": prog.feature_id,
        "family": prog.family.value,
        "program": _serialize_node(prog.root),
        "output": {
            "name": prog.output_name,
            "type": (prog.output_dtype.value if prog.output_dtype else "numeric"),
            "unit": (prog.output_unit.value if prog.output_unit else "unitless"),
        },
    }


def _parse_node(node: dict) -> ASTNode:
    if not isinstance(node, dict):
        raise GrammarError(f"node must be an object, got {type(node).__name__}")
    op_raw = node.get("op")
    if not op_raw:
        raise GrammarError("node missing op")
    try:
        op = Operator(op_raw)
    except ValueError:
        raise GrammarError(f"unknown operator: {op_raw}")

    sig = get_signature(op)

    if op == Operator.COL:
        return _parse_col(node)
    if op == Operator.CONST:
        return _parse_const(node)
    if op == Operator.TARGET_ENCODE:
        return _parse_target_encode(node)
    if sig.arity == -1:
        return _parse_variadic(node, op)
    if sig.arity == 1:
        return _parse_unary(node, op)
    if sig.arity == 2:
        return _parse_binary(node, op)

    raise GrammarError(f"unhandled operator: {op}")


def _parse_col(node: dict) -> ColRef:
    ci = node.get("col_index")
    if ci is None or not isinstance(ci, int):
        raise GrammarError("Col requires integer col_index")
    return ColRef(col_index=ci)


def _parse_const(node: dict) -> ConstVal:
    value = node.get("value")
    dtype_raw = node.get("dtype", "numeric")
    try:
        dtype = DataType(dtype_raw)
    except ValueError:
        raise GrammarError(f"unknown dtype: {dtype_raw}")
    return ConstVal(value=value, const_dtype=dtype)


def _parse_unary(node: dict, op: Operator) -> UnaryOp:
    child = node.get("child")
    if child is None:
        raise GrammarError(f"{op.value} requires child")
    return UnaryOp(op=op, child=_parse_node(child))


def _parse_binary(node: dict, op: Operator) -> BinaryOp:
    left = node.get("left")
    right = node.get("right")
    if left is None or right is None:
        raise GrammarError(f"{op.value} requires left and right")
    return BinaryOp(op=op, left=_parse_node(left), right=_parse_node(right))


def _parse_variadic(node: dict, op: Operator) -> VariadicOp:
    children = node.get("children")
    if not children or not isinstance(children, list) or len(children) < 2:
        raise GrammarError(f"{op.value} requires children list with >= 2 elements")
    return VariadicOp(op=op, children=[_parse_node(c) for c in children])


def _parse_target_encode(node: dict) -> TargetEncodeNode:
    cat_col = node.get("cat_col")
    label_col = node.get("label_col")
    if cat_col is None or label_col is None:
        raise GrammarError("TargetEncode requires cat_col and label_col")
    cat = _parse_node(cat_col)
    label = _parse_node(label_col)
    if not isinstance(cat, ColRef):
        raise GrammarError("TargetEncode cat_col must be Col")
    if not isinstance(label, ColRef):
        raise GrammarError("TargetEncode label_col must be Col")
    return TargetEncodeNode(cat_col=cat, label_col=label)


def _serialize_node(node: ASTNode) -> dict:
    if isinstance(node, ColRef):
        return {"op": "Col", "col_index": node.col_index}
    if isinstance(node, ConstVal):
        result: dict[str, Any] = {"op": "Const", "value": node.value}
        if node.const_dtype:
            result["dtype"] = node.const_dtype.value
        return result
    if isinstance(node, TargetEncodeNode):
        return {
            "op": "TargetEncode",
            "cat_col": _serialize_node(node.cat_col) if node.cat_col else {},
            "label_col": _serialize_node(node.label_col) if node.label_col else {},
        }
    if isinstance(node, VariadicOp):
        return {"op": node.op.value, "children": [_serialize_node(c) for c in node.children]}
    if isinstance(node, UnaryOp):
        return {"op": node.op.value, "child": _serialize_node(node.child) if node.child else {}}
    if isinstance(node, BinaryOp):
        return {
            "op": node.op.value,
            "left": _serialize_node(node.left) if node.left else {},
            "right": _serialize_node(node.right) if node.right else {},
        }
    raise GrammarError(f"unhandled AST node type: {type(node).__name__}")
