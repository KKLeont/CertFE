from __future__ import annotations

from .ast import (
    ASTNode, ColRef, ConstVal, UnaryOp, BinaryOp, VariadicOp,
    TargetEncodeNode, FeatureProgram,
)
from .operators import Operator, get_signature
from .types import DataType, Unit, _derive_unit, OUTPUT_TYPE_OVERRIDE, NUMERIC_ONLY_OPS, CATEGORICAL_ONLY_OPS
from .lineage import LineageNode, LineageGraph


class CompileError(Exception):
    """AST compilation failed — type error, missing field, or inconsistent derivation."""


def compile_program(
    prog: FeatureProgram,
    column_types: dict[int, DataType],
    column_units: dict[int, Unit] | None = None,
    target_col: int | None = None,
    split_col_name: str = "_split",
) -> tuple[FeatureProgram, LineageGraph]:
    """Type-check, annotate, and build lineage graph for a FeatureProgram.

    Returns the annotated program and its lineage graph.
    Raises CompileError on failure.
    """
    column_units = column_units or {}
    graph = LineageGraph()
    annotated_root, root_id = _compile_node(
        prog.root, column_types, column_units, target_col, graph,
    )
    prog.root = annotated_root
    graph.root_id = root_id

    # Derive output type/unit
    root_node = graph.nodes[root_id]
    if prog.output_dtype is None:
        prog.output_dtype = root_node.dtype
    if prog.output_unit is None:
        prog.output_unit = root_node.unit

    # Cross-check declared output vs compiled
    if prog.output_dtype and root_node.dtype and prog.output_dtype != root_node.dtype:
        raise CompileError(
            f"declared output type {prog.output_dtype.value} != "
            f"compiled {root_node.dtype.value}"
        )
    if prog.output_unit and root_node.unit and prog.output_unit != root_node.unit:
        raise CompileError(
            f"declared output unit {prog.output_unit.value} != "
            f"compiled {root_node.unit.value}"
        )

    return prog, graph


def _compile_node(
    node: ASTNode,
    column_types: dict[int, DataType],
    column_units: dict[int, Unit],
    target_col: int | None,
    graph: LineageGraph,
) -> tuple[ASTNode, str]:
    """Recursively compile a node. Returns (annotated_node, node_id_in_graph)."""

    if isinstance(node, ColRef):
        return _compile_col(node, column_types, column_units, graph)

    if isinstance(node, ConstVal):
        return _compile_const(node, graph)

    if isinstance(node, TargetEncodeNode):
        return _compile_target_encode(node, column_types, column_units, target_col, graph)

    if isinstance(node, VariadicOp):
        return _compile_variadic(node, column_types, column_units, target_col, graph)

    if isinstance(node, UnaryOp):
        return _compile_unary(node, column_types, column_units, target_col, graph)

    if isinstance(node, BinaryOp):
        return _compile_binary(node, column_types, column_units, target_col, graph)

    raise CompileError(f"unknown node type: {type(node).__name__}")


def _compile_col(
    node: ColRef, column_types: dict[int, DataType],
    column_units: dict[int, Unit], graph: LineageGraph,
) -> tuple[ColRef, str]:
    ci = node.col_index
    dtype = column_types.get(ci)
    if dtype is None:
        raise CompileError(f"column {ci} not in schema")
    node.dtype = dtype
    node.unit = column_units.get(ci, Unit.UNITLESS)
    ln = LineageNode(
        node_id="",
        op=Operator.COL,
        dtype=dtype,
        unit=node.unit,
        col_index=ci,
    )
    nid = graph.add_node(ln)
    return node, nid


def _compile_const(node: ConstVal, graph: LineageGraph) -> tuple[ConstVal, str]:
    dtype = node.const_dtype or DataType.NUMERIC
    node.dtype = dtype
    node.unit = Unit.UNITLESS
    ln = LineageNode(
        node_id="",
        op=Operator.CONST,
        dtype=dtype,
        unit=Unit.UNITLESS,
        const_value=node.value,
    )
    nid = graph.add_node(ln)
    return node, nid


def _compile_unary(
    node: UnaryOp, column_types: dict[int, DataType],
    column_units: dict[int, Unit], target_col: int | None,
    graph: LineageGraph,
) -> tuple[UnaryOp, str]:
    if node.child is None:
        raise CompileError(f"{node.op.value}: missing child")
    child, child_id = _compile_node(node.child, column_types, column_units, target_col, graph)

    op_name = node.op.value

    # Check numeric-only
    if op_name in NUMERIC_ONLY_OPS and child.dtype != DataType.NUMERIC:
        raise CompileError(
            f"{op_name}: expected numeric input, got {child.dtype.value}"
        )
    # Check categorical-only
    if op_name in CATEGORICAL_ONLY_OPS and child.dtype != DataType.CATEGORICAL:
        raise CompileError(
            f"{op_name}: expected categorical input, got {child.dtype.value}"
        )

    # Derive output type
    out_dtype = OUTPUT_TYPE_OVERRIDE.get(op_name, child.dtype)
    in_units = [child.unit] if child.unit else [Unit.UNITLESS]
    out_unit = _derive_unit(op_name, in_units)

    node.child = child
    node.dtype = out_dtype
    node.unit = out_unit

    ln = LineageNode(
        node_id="",
        op=node.op,
        dtype=out_dtype,
        unit=out_unit,
        children=[child_id],
    )
    nid = graph.add_node(ln)
    return node, nid


def _compile_binary(
    node: BinaryOp, column_types: dict[int, DataType],
    column_units: dict[int, Unit], target_col: int | None,
    graph: LineageGraph,
) -> tuple[BinaryOp, str]:
    if node.left is None or node.right is None:
        raise CompileError(f"{node.op.value}: missing left or right child")
    left, left_id = _compile_node(node.left, column_types, column_units, target_col, graph)
    right, right_id = _compile_node(node.right, column_types, column_units, target_col, graph)

    op_name = node.op.value

    # Type checks
    if op_name in NUMERIC_ONLY_OPS:
        if left.dtype != DataType.NUMERIC or right.dtype != DataType.NUMERIC:
            raise CompileError(f"{op_name}: both inputs must be numeric")

    if op_name in CATEGORICAL_ONLY_OPS:
        if left.dtype != DataType.CATEGORICAL and right.dtype != DataType.CATEGORICAL:
            raise CompileError(f"{op_name}: at least one input must be categorical")

    # Unit checks for specific ops
    if op_name in ("Add", "Subtract"):
        if left.unit != right.unit:
            raise CompileError(
                f"{op_name}: unit mismatch — {left.unit.value} vs {right.unit.value}"
            )

    node.left = left
    node.right = right
    out_dtype = OUTPUT_TYPE_OVERRIDE.get(op_name, DataType.NUMERIC)
    out_unit = _derive_unit(op_name, [left.unit or Unit.UNITLESS, right.unit or Unit.UNITLESS])
    node.dtype = out_dtype
    node.unit = out_unit

    ln = LineageNode(
        node_id="",
        op=node.op,
        dtype=out_dtype,
        unit=out_unit,
        children=[left_id, right_id],
    )
    nid = graph.add_node(ln)
    return node, nid


def _compile_variadic(
    node: VariadicOp, column_types: dict[int, DataType],
    column_units: dict[int, Unit], target_col: int | None,
    graph: LineageGraph,
) -> tuple[VariadicOp, str]:
    if len(node.children) < 2:
        raise CompileError(f"{node.op.value}: need >= 2 children")

    compiled: list[ASTNode] = []
    child_ids: list[str] = []
    first_unit: Unit | None = None
    for child in node.children:
        c_annot, cid = _compile_node(child, column_types, column_units, target_col, graph)
        if c_annot.dtype != DataType.NUMERIC:
            raise CompileError(
                f"{node.op.value}: all children must be numeric, "
                f"got {c_annot.dtype.value}"
            )
        if first_unit is None:
            first_unit = c_annot.unit
        compiled.append(c_annot)
        child_ids.append(cid)

    node.children = compiled
    out_unit = _derive_unit(node.op.value, [first_unit or Unit.UNITLESS])
    node.dtype = DataType.NUMERIC
    node.unit = out_unit

    ln = LineageNode(
        node_id="",
        op=node.op,
        dtype=DataType.NUMERIC,
        unit=out_unit,
        children=child_ids,
    )
    nid = graph.add_node(ln)
    return node, nid


def _compile_target_encode(
    node: TargetEncodeNode, column_types: dict[int, DataType],
    column_units: dict[int, Unit], target_col: int | None,
    graph: LineageGraph,
) -> tuple[TargetEncodeNode, str]:
    if node.cat_col is None or node.label_col is None:
        raise CompileError("TargetEncode: missing cat_col or label_col")

    cat, cat_id = _compile_node(node.cat_col, column_types, column_units, target_col, graph)
    label, label_id = _compile_node(node.label_col, column_types, column_units, target_col, graph)

    if cat.dtype != DataType.CATEGORICAL:
        raise CompileError(f"TargetEncode: cat_col must be categorical, got {cat.dtype.value}")

    if target_col is not None and node.label_col.col_index != target_col:
        raise CompileError(
            f"TargetEncode: label_col must be the target column ({target_col}), "
            f"got {node.label_col.col_index}"
        )

    node.cat_col = cat
    node.label_col = label
    node.dtype = DataType.NUMERIC
    node.unit = Unit.PROBABILITY

    ln = LineageNode(
        node_id="",
        op=Operator.TARGET_ENCODE,
        dtype=DataType.NUMERIC,
        unit=Unit.PROBABILITY,
        children=[cat_id, label_id],
    )
    nid = graph.add_node(ln)
    return node, nid
