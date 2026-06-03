from enum import Enum


class DataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Unit(str, Enum):
    UNITLESS = "unitless"
    RATIO = "ratio"
    COUNT = "count"
    LOG_COUNT = "log_count"
    SQUARED = "squared"
    PROBABILITY = "probability"
    # extensible


def _derive_unit(op: str, input_units: list[Unit]) -> Unit:
    """Derive output unit from operator and input units.

    Centralised here so both compile.py and the verifier share the same rules.
    """
    if op in ("Log1p",):
        return Unit.COUNT if input_units[0] == Unit.COUNT else Unit.UNITLESS
    if op in ("Sqrt",):
        return Unit.UNITLESS
    if op in ("Abs", "Clip", "ZScore", "MinMaxScale"):
        return input_units[0]
    if op in ("Square",):
        return Unit.SQUARED
    if op in ("Reciprocal",):
        return Unit.UNITLESS
    if op in ("Add", "Subtract"):
        return input_units[0]  # same-unit enforced by verifier
    if op in ("Multiply",):
        return Unit.UNITLESS  # conservative
    if op in ("Divide", "Ratio"):
        return Unit.RATIO if op == "Ratio" else Unit.UNITLESS
    if op in ("FrequencyEncode",):
        return Unit.COUNT
    if op in ("TargetEncode",):
        return Unit.PROBABILITY
    if op in ("IsNull", "IsEqual", "OneHotSelect"):
        return Unit.UNITLESS
    if op in ("Coalesce",):
        return input_units[0]
    if op in ("Cross",):
        return Unit.UNITLESS
    if op in ("QuantileBucketize",):
        return Unit.UNITLESS
    if op in ("RowMean", "RowMax", "RowMin", "RowSum", "RowStd"):
        return input_units[0]
    return Unit.UNITLESS


# Operators that require numeric input (used by C4 verifier)
NUMERIC_ONLY_OPS = frozenset({
    "Log1p", "Sqrt", "Abs", "Square", "Reciprocal", "Clip",
    "ZScore", "MinMaxScale", "QuantileBucketize",
    "Add", "Subtract", "Multiply", "Divide", "Ratio",
    "RowMean", "RowMax", "RowMin", "RowSum", "RowStd",
})

# Operators that require categorical input
CATEGORICAL_ONLY_OPS = frozenset({
    "FrequencyEncode", "Cross", "OneHotSelect",
})

# Output type overrides (operator -> forced output DataType)
OUTPUT_TYPE_OVERRIDE = {
    "QuantileBucketize": DataType.CATEGORICAL,
    "Cross": DataType.CATEGORICAL,
    "IsNull": DataType.BOOLEAN,
    "IsEqual": DataType.BOOLEAN,
    "OneHotSelect": DataType.BOOLEAN,
    "TargetEncode": DataType.NUMERIC,
    "FrequencyEncode": DataType.NUMERIC,
}
