from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .ast import (
    ASTNode, ColRef, ConstVal, UnaryOp, BinaryOp, VariadicOp,
    TargetEncodeNode, FeatureProgram,
)
from .operators import Operator
from .types import DataType


class ExecutionError(Exception):
    """Program evaluation failed."""


@dataclass
class FitStats:
    """Parameters learned from training data for fit-ops.

    Keys are node_ids from the lineage graph (for fit-ops) or column indices
    (for raw-column fit-ops like FrequencyEncode on a bare Col).
    """
    # ZScore / MinMaxScale: {node_id: (mean, std) or (min, max)}
    scale_params: dict[str, tuple[float, float]] = field(default_factory=dict)
    # FrequencyEncode: {node_id: {category: count}}
    freq_maps: dict[str, dict[object, float]] = field(default_factory=dict)
    # TargetEncode: {node_id: {category: mean_target}}
    target_enc_maps: dict[str, dict[object, float]] = field(default_factory=dict)
    # QuantileBucketize: {node_id: [boundary_1, boundary_2, ...]}
    quantile_boundaries: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class EvaluationContext:
    """Runtime context for program evaluation."""
    df: pd.DataFrame
    split_col: str | None = "_split"
    fit_split: str = "train"
    fit_stats: FitStats | None = None
    node_cache: dict[str, pd.Series] = field(default_factory=dict)

    def get_split_mask(self, split: str) -> pd.Series:
        if self.split_col is None or self.split_col not in self.df.columns:
            return pd.Series(True, index=self.df.index)
        return self.df[self.split_col] == split


def evaluate(prog: FeatureProgram, ctx: EvaluationContext) -> pd.Series:
    """Evaluate a compiled FeatureProgram on a DataFrame."""
    ctx.node_cache = {}
    return _eval(prog.root, prog.feature_id, ctx)


def fit(prog: FeatureProgram, df_train: pd.DataFrame, col_types: dict[int, DataType]) -> FitStats:
    """Fit all fit-ops in a program on training data only."""
    stats = FitStats()
    ctx = EvaluationContext(df=df_train, fit_stats=stats)
    ctx.node_cache = {}
    _eval(prog.root, prog.feature_id, ctx)
    return stats


def _eval(node: ASTNode, node_key: str, ctx: EvaluationContext) -> pd.Series:
    if node_key in ctx.node_cache:
        return ctx.node_cache[node_key]

    if isinstance(node, ColRef):
        result = _eval_col(node, ctx)
    elif isinstance(node, ConstVal):
        result = _eval_const(node, ctx)
    elif isinstance(node, TargetEncodeNode):
        result = _eval_target_encode(node, node_key, ctx)
    elif isinstance(node, VariadicOp):
        result = _eval_variadic(node, node_key, ctx)
    elif isinstance(node, UnaryOp):
        result = _eval_unary(node, node_key, ctx)
    elif isinstance(node, BinaryOp):
        result = _eval_binary(node, node_key, ctx)
    else:
        raise ExecutionError(f"unknown node type: {type(node).__name__}")

    ctx.node_cache[node_key] = result
    return result


def _eval_col(node: ColRef, ctx: EvaluationContext) -> pd.Series:
    if node.col_index not in range(len(ctx.df.columns)):
        raise ExecutionError(f"column index {node.col_index} out of range")
    col = ctx.df.iloc[:, node.col_index]
    col = pd.to_numeric(col, errors="coerce") if node.dtype == DataType.NUMERIC else col.astype(str)
    return col


def _eval_const(node: ConstVal, ctx: EvaluationContext) -> pd.Series:
    return pd.Series(node.value, index=ctx.df.index)


def _eval_unary(node: UnaryOp, node_key: str, ctx: EvaluationContext) -> pd.Series:
    child_key = f"{node_key}/child"
    x = _eval(node.child, child_key, ctx)

    op = node.op
    if op == Operator.LOG1P:
        return np.log1p(pd.to_numeric(x, errors="coerce"))
    if op == Operator.SQRT:
        return np.sqrt(pd.to_numeric(x, errors="coerce").clip(lower=0))
    if op == Operator.ABS:
        return pd.to_numeric(x, errors="coerce").abs()
    if op == Operator.SQUARE:
        return pd.to_numeric(x, errors="coerce") ** 2
    if op == Operator.RECIPROCAL:
        s = pd.to_numeric(x, errors="coerce")
        return 1.0 / s.where(s != 0, other=np.nan)
    if op == Operator.CLIP:
        s = pd.to_numeric(x, errors="coerce")
        lo = float(node.child.const_value) if hasattr(node.child, 'const_value') else s.quantile(0.01)
        hi = float(node.child.const_value) if hasattr(node.child, 'const_value') else s.quantile(0.99)
        return s.clip(lower=lo, upper=hi)
    if op == Operator.ZSCORE:
        return _fit_eval_scale(node_key, x, ctx, method="zscore")
    if op == Operator.MINMAX_SCALE:
        return _fit_eval_scale(node_key, x, ctx, method="minmax")
    if op == Operator.QUANTILE_BUCKETIZE:
        return _fit_eval_quantile(node_key, x, ctx)
    if op == Operator.FREQUENCY_ENCODE:
        return _fit_eval_freq(node_key, x, ctx)
    if op == Operator.IS_NULL:
        return x.isna().astype(float)
    if op == Operator.COALESCE:
        # Coalesce is binary in AST; handled in _eval_binary
        pass

    raise ExecutionError(f"unhandled unary op: {op}")


def _eval_binary(node: BinaryOp, node_key: str, ctx: EvaluationContext) -> pd.Series:
    left_key = f"{node_key}/L"
    right_key = f"{node_key}/R"
    # Evaluate left first; for Coalesce, short-circuit
    if node.op == Operator.COALESCE:
        # binary form: Coalesce(col, const)
        left = _eval(node.left, left_key, ctx)
        # right should be a ConstVal
        fill_val = None
        if isinstance(node.right, ConstVal):
            fill_val = node.right.value
        return left.fillna(fill_val)

    left = _eval(node.left, left_key, ctx)
    right = _eval(node.right, right_key, ctx)

    op = node.op
    if op == Operator.ADD:
        return pd.to_numeric(left, errors="coerce") + pd.to_numeric(right, errors="coerce")
    if op == Operator.SUBTRACT:
        return pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")
    if op == Operator.MULTIPLY:
        return pd.to_numeric(left, errors="coerce") * pd.to_numeric(right, errors="coerce")
    if op == Operator.DIVIDE:
        num = pd.to_numeric(left, errors="coerce")
        den = pd.to_numeric(right, errors="coerce")
        return num / den.where(den != 0, other=np.nan)
    if op == Operator.RATIO:
        num = pd.to_numeric(left, errors="coerce")
        den = pd.to_numeric(right, errors="coerce")
        return num / den.where(den != 0, other=np.nan)
    if op == Operator.CROSS:
        return left.astype(str) + "_x_" + right.astype(str)
    if op == Operator.IS_EQUAL:
        if isinstance(node.right, ConstVal):
            return (left.astype(str) == str(node.right.value)).astype(float)
        return (left.astype(str) == right.astype(str)).astype(float)
    if op == Operator.ONE_HOT_SELECT:
        if isinstance(node.right, ConstVal):
            return (left.astype(str) == str(node.right.value)).astype(float)
        return (left.astype(str) == right.astype(str)).astype(float)

    raise ExecutionError(f"unhandled binary op: {op}")


def _eval_variadic(node: VariadicOp, node_key: str, ctx: EvaluationContext) -> pd.Series:
    series_list = []
    for i, child in enumerate(node.children):
        child_key = f"{node_key}/c{i}"
        s = _eval(child, child_key, ctx)
        series_list.append(pd.to_numeric(s, errors="coerce"))

    df = pd.concat(series_list, axis=1)
    op = node.op
    if op == Operator.ROW_MEAN:
        return df.mean(axis=1)
    if op == Operator.ROW_MAX:
        return df.max(axis=1)
    if op == Operator.ROW_MIN:
        return df.min(axis=1)
    if op == Operator.ROW_SUM:
        return df.sum(axis=1)
    if op == Operator.ROW_STD:
        return df.std(axis=1)

    raise ExecutionError(f"unhandled variadic op: {op}")


def _eval_target_encode(node: TargetEncodeNode, node_key: str, ctx: EvaluationContext) -> pd.Series:
    cat_key = f"{node_key}/cat"
    label_key = f"{node_key}/label"
    cat = _eval(node.cat_col, cat_key, ctx).astype(str)
    label = _eval(node.label_col, label_key, ctx)

    train_mask = ctx.get_split_mask("train")

    if ctx.fit_stats is not None and node_key in ctx.fit_stats.target_enc_maps:
        enc = ctx.fit_stats.target_enc_maps[node_key]
        return cat.map(enc)

    # Fit on train split
    train_cat = cat[train_mask]
    train_label = pd.to_numeric(label[train_mask], errors="coerce")
    enc = train_label.groupby(train_cat).mean().to_dict()

    if ctx.fit_stats is not None:
        ctx.fit_stats.target_enc_maps[node_key] = enc

    return cat.map(enc)


def _fit_eval_scale(node_key: str, x: pd.Series, ctx: EvaluationContext, method: str) -> pd.Series:
    train_mask = ctx.get_split_mask("train")

    if ctx.fit_stats is not None and node_key in ctx.fit_stats.scale_params:
        mu, sigma = ctx.fit_stats.scale_params[node_key]
        s = pd.to_numeric(x, errors="coerce")
        return (s - mu) / sigma if method == "zscore" else (s - sigma) / (mu - sigma) if mu != sigma else s

    train_x = pd.to_numeric(x[train_mask], errors="coerce").dropna()
    if method == "zscore":
        mu, sigma = train_x.mean(), train_x.std()
        if sigma == 0:
            sigma = 1.0
    else:
        mu, sigma = train_x.max(), train_x.min()

    if ctx.fit_stats is not None:
        ctx.fit_stats.scale_params[node_key] = (float(mu), float(sigma))

    s = pd.to_numeric(x, errors="coerce")
    if method == "zscore":
        return (s - mu) / sigma
    return (s - sigma) / (mu - sigma) if mu != sigma else s


def _fit_eval_freq(node_key: str, x: pd.Series, ctx: EvaluationContext) -> pd.Series:
    x_str = x.astype(str)
    train_mask = ctx.get_split_mask("train")

    if ctx.fit_stats is not None and node_key in ctx.fit_stats.freq_maps:
        freq = ctx.fit_stats.freq_maps[node_key]
        return x_str.map(freq)

    train_x = x_str[train_mask]
    freq = train_x.value_counts().to_dict()
    if ctx.fit_stats is not None:
        ctx.fit_stats.freq_maps[node_key] = freq
    return x_str.map(freq)


def _fit_eval_quantile(node_key: str, x: pd.Series, ctx: EvaluationContext) -> pd.Series:
    train_mask = ctx.get_split_mask("train")
    n_bins = 10

    s = pd.to_numeric(x, errors="coerce")

    if ctx.fit_stats is not None and node_key in ctx.fit_stats.quantile_boundaries:
        boundaries = ctx.fit_stats.quantile_boundaries[node_key]
        return pd.cut(s, bins=[-np.inf] + boundaries + [np.inf], labels=False)

    train_x = s[train_mask].dropna()
    boundaries = [float(train_x.quantile(q)) for q in np.linspace(0, 1, n_bins + 1)[1:-1]]
    if ctx.fit_stats is not None:
        ctx.fit_stats.quantile_boundaries[node_key] = boundaries

    return pd.cut(s, bins=[-np.inf] + boundaries + [np.inf], labels=False)
