from dataclasses import dataclass, field

from certfe.dsl.types import DataType, Unit


@dataclass
class CertificatePolicy:
    """Which certificates are active for a given dataset class.

    Controls the active set: {C1, C2, C4, C5, C6} for IID single-table;
    add C3 when t* is available.
    """
    c1_schema_validity: bool = True
    c2_no_leakage: bool = True
    c3_temporal_validity: bool = False  # only when t* != bot
    c4_type_unit_safety: bool = True
    c5_train_only_fitting: bool = True
    c6_lineage: bool = True

    # Schema data needed by verifiers
    column_types: dict[int, DataType] = field(default_factory=dict)
    column_units: dict[int, Unit] = field(default_factory=dict)
    target_col: int | None = None
    forbidden_cols: set[int] = field(default_factory=set)
    prediction_time: str | None = None  # t*; None means n/a (IID datasets)
    split_col: str = "_split"
    train_split_label: str = "train"
