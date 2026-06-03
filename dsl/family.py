from enum import Enum


class FeatureFamily(str, Enum):
    NONLINEAR_TRANSFORM = "nonlinear_transform"
    NORMALIZATION = "normalization"
    BINNING = "binning"
    ARITHMETIC = "arithmetic"
    ROWWISE_STATS = "rowwise_stats"
    CATEGORICAL_ENCODING = "categorical_encoding"
    CATEGORICAL_INTERACTION = "categorical_interaction"
    MISSINGNESS = "missingness"
    TEMPORAL_AGG = "temporal_aggregation"
    TEMPORAL_TREND = "temporal_trend"
    VOLATILITY = "volatility"
    ENTITY_AGG = "entity_aggregation"


# Families applicable on IID single-table (no t*, no entity key)
IID_FAMILIES = frozenset({
    FeatureFamily.NONLINEAR_TRANSFORM,
    FeatureFamily.NORMALIZATION,
    FeatureFamily.BINNING,
    FeatureFamily.ARITHMETIC,
    FeatureFamily.ROWWISE_STATS,
    FeatureFamily.CATEGORICAL_ENCODING,
    FeatureFamily.CATEGORICAL_INTERACTION,
    FeatureFamily.MISSINGNESS,
})

# Families that require t* (temporal anchor)
TEMPORAL_FAMILIES = frozenset({
    FeatureFamily.TEMPORAL_AGG,
    FeatureFamily.TEMPORAL_TREND,
    FeatureFamily.VOLATILITY,
})

# Families that require entity key / multi-table
RELATIONAL_FAMILIES = frozenset({
    FeatureFamily.ENTITY_AGG,
})
