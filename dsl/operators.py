from dataclasses import dataclass
from enum import Enum

from .types import DataType
from .family import FeatureFamily


class Operator(str, Enum):
    # Leaf
    COL = "Col"
    CONST = "Const"
    # Unary — nonlinear transform
    LOG1P = "Log1p"
    SQRT = "Sqrt"
    ABS = "Abs"
    SQUARE = "Square"
    RECIPROCAL = "Reciprocal"
    # Unary — normalization
    CLIP = "Clip"
    ZSCORE = "ZScore"
    MINMAX_SCALE = "MinMaxScale"
    # Unary — binning
    QUANTILE_BUCKETIZE = "QuantileBucketize"
    # Binary — arithmetic
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    RATIO = "Ratio"
    # Categorical
    FREQUENCY_ENCODE = "FrequencyEncode"
    TARGET_ENCODE = "TargetEncode"
    CROSS = "Cross"
    IS_EQUAL = "IsEqual"
    ONE_HOT_SELECT = "OneHotSelect"
    # Missing / logic
    IS_NULL = "IsNull"
    COALESCE = "Coalesce"
    # Row-wise
    ROW_MEAN = "RowMean"
    ROW_MAX = "RowMax"
    ROW_MIN = "RowMin"
    ROW_SUM = "RowSum"
    ROW_STD = "RowStd"
    # Temporal (reserved for panel / leakage injection)
    WINDOW_AGG = "WindowAgg"
    LAG = "Lag"
    TREND = "Trend"
    TIME_SINCE = "TimeSince"
    ROLLING_MEAN = "RollingMean"
    ROLLING_STD = "RollingStd"
    # Entity / relational (reserved)
    GROUP_BY_AGG = "GroupByAgg"
    ENTITY_JOIN = "EntityJoin"


@dataclass(frozen=True)
class OpSignature:
    op: Operator
    family: FeatureFamily
    arity: int  # positive; use arity=0 for leaf, arity=-1 for variadic
    is_fit_op: bool = False
    y_allowed: bool = False  # may reference target column in a dedicated slot


_ = FeatureFamily  # shorthand

OPERATOR_REGISTRY: dict[Operator, OpSignature] = {
    # Leaf
    Operator.COL:    OpSignature(Operator.COL,    _.MISSINGNESS,            arity=0),
    Operator.CONST:  OpSignature(Operator.CONST,  _.MISSINGNESS,            arity=0),

    # Nonlinear transform
    Operator.LOG1P:       OpSignature(Operator.LOG1P,       _.NONLINEAR_TRANSFORM, arity=1),
    Operator.SQRT:        OpSignature(Operator.SQRT,        _.NONLINEAR_TRANSFORM, arity=1),
    Operator.ABS:         OpSignature(Operator.ABS,         _.NONLINEAR_TRANSFORM, arity=1),
    Operator.SQUARE:      OpSignature(Operator.SQUARE,      _.NONLINEAR_TRANSFORM, arity=1),
    Operator.RECIPROCAL:  OpSignature(Operator.RECIPROCAL,  _.NONLINEAR_TRANSFORM, arity=1),

    # Normalization
    Operator.CLIP:          OpSignature(Operator.CLIP,          _.NORMALIZATION, arity=1),
    Operator.ZSCORE:        OpSignature(Operator.ZSCORE,        _.NORMALIZATION, arity=1, is_fit_op=True),
    Operator.MINMAX_SCALE:  OpSignature(Operator.MINMAX_SCALE,  _.NORMALIZATION, arity=1, is_fit_op=True),

    # Binning
    Operator.QUANTILE_BUCKETIZE: OpSignature(Operator.QUANTILE_BUCKETIZE, _.BINNING, arity=1, is_fit_op=True),

    # Arithmetic
    Operator.ADD:       OpSignature(Operator.ADD,       _.ARITHMETIC, arity=2),
    Operator.SUBTRACT:  OpSignature(Operator.SUBTRACT,  _.ARITHMETIC, arity=2),
    Operator.MULTIPLY:  OpSignature(Operator.MULTIPLY,  _.ARITHMETIC, arity=2),
    Operator.DIVIDE:    OpSignature(Operator.DIVIDE,    _.ARITHMETIC, arity=2),
    Operator.RATIO:     OpSignature(Operator.RATIO,     _.ARITHMETIC, arity=2),

    # Categorical encoding
    Operator.FREQUENCY_ENCODE: OpSignature(Operator.FREQUENCY_ENCODE, _.CATEGORICAL_ENCODING, arity=1, is_fit_op=True),
    Operator.TARGET_ENCODE:    OpSignature(Operator.TARGET_ENCODE,    _.CATEGORICAL_ENCODING, arity=2, is_fit_op=True, y_allowed=True),

    # Categorical interaction
    Operator.CROSS:          OpSignature(Operator.CROSS,          _.CATEGORICAL_INTERACTION, arity=2),
    Operator.IS_EQUAL:       OpSignature(Operator.IS_EQUAL,       _.CATEGORICAL_INTERACTION, arity=2),
    Operator.ONE_HOT_SELECT: OpSignature(Operator.ONE_HOT_SELECT, _.CATEGORICAL_INTERACTION, arity=2),

    # Missing / logic
    Operator.IS_NULL:  OpSignature(Operator.IS_NULL,  _.MISSINGNESS, arity=1),
    Operator.COALESCE: OpSignature(Operator.COALESCE, _.MISSINGNESS, arity=2),

    # Row-wise (variadic)
    Operator.ROW_MEAN: OpSignature(Operator.ROW_MEAN, _.ROWWISE_STATS, arity=-1),
    Operator.ROW_MAX:  OpSignature(Operator.ROW_MAX,  _.ROWWISE_STATS, arity=-1),
    Operator.ROW_MIN:  OpSignature(Operator.ROW_MIN,  _.ROWWISE_STATS, arity=-1),
    Operator.ROW_SUM:  OpSignature(Operator.ROW_SUM,  _.ROWWISE_STATS, arity=-1),
    Operator.ROW_STD:  OpSignature(Operator.ROW_STD,  _.ROWWISE_STATS, arity=-1),

    # Temporal (reserved)
    Operator.WINDOW_AGG:    OpSignature(Operator.WINDOW_AGG,    _.TEMPORAL_AGG,   arity=1),
    Operator.LAG:           OpSignature(Operator.LAG,           _.TEMPORAL_TREND, arity=1),
    Operator.TREND:         OpSignature(Operator.TREND,         _.TEMPORAL_TREND, arity=2),
    Operator.TIME_SINCE:    OpSignature(Operator.TIME_SINCE,    _.TEMPORAL_TREND, arity=1),
    Operator.ROLLING_MEAN:  OpSignature(Operator.ROLLING_MEAN,  _.VOLATILITY,     arity=1),
    Operator.ROLLING_STD:   OpSignature(Operator.ROLLING_STD,   _.VOLATILITY,     arity=1),

    # Entity / relational (reserved)
    Operator.GROUP_BY_AGG: OpSignature(Operator.GROUP_BY_AGG, _.ENTITY_AGG, arity=2),
    Operator.ENTITY_JOIN:  OpSignature(Operator.ENTITY_JOIN,  _.ENTITY_AGG, arity=2),
}

FIT_OPS: frozenset[Operator] = frozenset({
    op for op, sig in OPERATOR_REGISTRY.items() if sig.is_fit_op
})

Y_ALLOWED_OPS: frozenset[Operator] = frozenset({
    op for op, sig in OPERATOR_REGISTRY.items() if sig.y_allowed
})


def get_signature(op: Operator) -> OpSignature:
    return OPERATOR_REGISTRY[op]
