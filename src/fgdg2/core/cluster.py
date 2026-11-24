
from .node import Node


class Cluster:
    def __init__(self, nodes):
        ids = [n.id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"node ids must be unique, got duplicates in {ids}")
        self.nodes: list[Node] = list(nodes)
        self._by_id: dict = {n.id: n for n in self.nodes}

    def node_by_id(self, node_id) -> Node:
        return self._by_id[node_id]

    def add_node(self, node: Node) -> None:
        """Bring a node online (elastic scale-up). Ids must be fresh per
        incarnation — never recycle an evicted id while receipts may be live."""
        if node.id in self._by_id:
            raise ValueError(f"duplicate node id {node.id}")
        self.nodes.append(node)
        self._by_id[node.id] = node

    def remove_node(self, node_id) -> Node:
        """Take a node offline (elastic scale-down). Caller must only evict
        empty nodes — live receipts resolve by id and would KeyError."""
        node = self._by_id.pop(node_id)
        self.nodes.remove(node)
        return node

    @property
    def total_gpus(self) -> int:
        return sum(n.num_gpus for n in self.nodes)

    @property
    def total_free_slots(self) -> int:
        return sum(n.free_slots for n in self.nodes)

    @property
    def total_used_gpus(self) -> float:
        return sum(n.num_gpus - n.gpu_free.sum() for n in self.nodes)

    def feasible_nodes(self, cls):
        return [n for n in self.nodes if n.fits(cls)]

    def place(self, node: Node, cls, gpu_idx=None):
        if self._by_id.get(node.id) is not node:
            raise ValueError(f"node {node.id} does not belong to this cluster")
        return node.place(cls, gpu_idx=gpu_idx)

    def copy(self) -> "Cluster":
        """Deep copy of the nodes (same ids) for dry-run gang placement."""
        return Cluster([n.copy() for n in self.nodes])

    def __repr__(self):
        return f"Cluster(nodes={len(self.nodes)}, free={self.total_free_slots}/{self.total_gpus})"
