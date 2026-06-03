from __future__ import annotations

from dataclasses import dataclass, field

from .types import DataType, Unit
from .operators import Operator
from .family import FeatureFamily


@dataclass
class ASTNode:
    op: Operator | None = None       # set by subclass __post_init__
    dtype: DataType | None = None    # filled by compiler
    unit: Unit | None = None          # filled by compiler
    col_indices: tuple[int, ...] = ()  # leaf columns in subtree (filled by compiler)


@dataclass
class ColRef(ASTNode):
    """Reference to a column by index."""
    col_index: int = -1

    def __post_init__(self):
        self.op = Operator.COL
        self.col_indices = (self.col_index,)


@dataclass
class ConstVal(ASTNode):
    """Literal constant."""
    value: int | float | str | None = None
    const_dtype: DataType | None = None

    def __post_init__(self):
        self.op = Operator.CONST
        if self.const_dtype is not None:
            self.dtype = self.const_dtype
        self.col_indices = ()


@dataclass
class UnaryOp(ASTNode):
    """Single-child operator node."""
    child: ASTNode | None = None

    def __post_init__(self):
        if self.child is not None:
            self.col_indices = self.child.col_indices


@dataclass
class BinaryOp(ASTNode):
    """Two-child operator node."""
    left: ASTNode | None = None
    right: ASTNode | None = None

    def __post_init__(self):
        self.col_indices = ()
        if self.left is not None and self.right is not None:
            self.col_indices = tuple(set(self.left.col_indices + self.right.col_indices))


@dataclass
class VariadicOp(ASTNode):
    """N-child operator node (row-wise stats)."""
    children: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        indices: list[int] = []
        for c in self.children:
            indices.extend(c.col_indices)
        self.col_indices = tuple(set(indices))


@dataclass
class TargetEncodeNode(ASTNode):
    """TargetEncode with a dedicated label slot (y-allowed)."""
    cat_col: ColRef | None = None
    label_col: ColRef | None = None

    def __post_init__(self):
        self.op = Operator.TARGET_ENCODE
        indices = []
        if self.cat_col is not None:
            indices.extend(self.cat_col.col_indices)
        if self.label_col is not None:
            indices.extend(self.label_col.col_indices)
        self.col_indices = tuple(set(indices))


@dataclass
class FeatureProgram:
    feature_id: str
    family: FeatureFamily
    root: ASTNode
    output_name: str = ""
    output_dtype: DataType | None = None
    output_unit: Unit | None = None
