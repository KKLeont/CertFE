from __future__ import annotations

from dataclasses import dataclass, field

from .operators import Operator
from .types import DataType, Unit


@dataclass
class LineageNode:
    node_id: str
    op: Operator
    dtype: DataType | None = None
    unit: Unit | None = None
    source_split: str | None = None  # "train", "val", "test", or None
    time_origin: str | None = None    # time window expression, or None
    children: list[str] = field(default_factory=list)

    # For leaf nodes
    col_index: int | None = None
    const_value: object = None


@dataclass
class LineageGraph:
    nodes: dict[str, LineageNode] = field(default_factory=dict)
    root_id: str = ""
    _node_counter: int = field(default=0, init=False)

    def add_node(self, node: LineageNode) -> str:
        nid = node.node_id or f"n{self._node_counter}"
        self._node_counter += 1
        node.node_id = nid
        self.nodes[nid] = node
        return nid

    def ancestors(self, node_id: str) -> set[str]:
        result: set[str] = set()
        if node_id not in self.nodes:
            return result
        for child_id in self.nodes[node_id].children:
            result.add(child_id)
            result.update(self.ancestors(child_id))
        return result

    def leaf_columns(self) -> set[int]:
        cols: set[int] = set()
        for node in self.nodes.values():
            if node.col_index is not None:
                cols.add(node.col_index)
        return cols

    def nodes_by_op(self, op: Operator) -> list[LineageNode]:
        return [n for n in self.nodes.values() if n.op == op]
