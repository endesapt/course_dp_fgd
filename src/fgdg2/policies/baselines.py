import numpy as np

from ..core.cluster import Cluster
from ..core.node import Node
from ..core.rmath import TOL


def _scalar_gpu_capacity(node: Node) -> float:
    """R^GPU scalarization: fully-free count + largest partial share (paper 3.1)."""
    r = node.gpu_free
    if len(r) == 0:
        return 0.0
    partial = r[~np.isclose(r, 1.0, atol=TOL)]
    p = float(partial.max()) if len(partial) else 0.0
    return float(np.count_nonzero(r >= 1.0 - TOL)) + p


class BestFitPolicy:
    """Least remaining resources, weighted sum over normalized dims (paper Baselines 1).

    Node-level: argmin of normalized remaining CPU + scalarized GPU after the
    task fits. Within node, fractional shares go to the GPU with the least
    remaining that still fits (paper extension 2).
    """

    def __init__(self, max_cpu_milli: float, max_gpu: float, cpu_weight: float = 1.0):
        """cpu_weight scales the CPU term of the residual (1.0 = paper BestFit;
        0.0 = GPU-only BestFit, the `atomic_bestfit_gpu` ablation)."""
        self.max_cpu = float(max_cpu_milli)
        self.max_gpu = float(max_gpu)
        self.cpu_weight = float(cpu_weight)

    @classmethod
    def from_cluster(cls, cluster: Cluster, cpu_weight: float = 1.0):
        return cls(max(n.cpu_milli for n in cluster.nodes),
                   max(n.num_gpus for n in cluster.nodes), cpu_weight=cpu_weight)

    def _residual(self, node: Node, cls) -> float:
        r = node.gpu_free
        if len(r):
            r2 = r.copy()
            if cls.gpu_demand > 0.0:
                if cls.gpu_demand >= 1.0:
                    full = np.flatnonzero(r2 >= 1.0 - TOL)
                    r2[full[: int(cls.gpu_demand)]] = 0.0
                else:
                    fit_idx = np.flatnonzero(r2 >= cls.gpu_demand - TOL)
                    g = int(fit_idx[np.argmin(r2[fit_idx])])
                    r2[g] -= cls.gpu_demand
            partial = r2[~np.isclose(r2, 1.0, atol=TOL)]
            gpu_left = np.count_nonzero(r2 >= 1.0 - TOL) + (float(partial.max()) if len(partial) else 0.0)
        else:
            gpu_left = 0.0
        cpu_left = max(node.cpu_free - cls.cpu_milli, 0)
        return gpu_left / self.max_gpu + self.cpu_weight * cpu_left / self.max_cpu

    def decide(self, cluster: Cluster, cls):
        best = None
        best_key = None
        for node in cluster.nodes:
            if not node.fits(cls):
                continue
            key = self._residual(node, cls)
            if best_key is None or key < best_key:
                best_key, best = key, node
        if best is None:
            return (None, None)
        gpu_idx = self._gpu_in_node(best, cls)
        return (best, gpu_idx)

    def _gpu_in_node(self, node: Node, cls) -> int | None:
        d = cls.gpu_demand
        if d == 0.0 or d >= 1.0:
            return None
        r = node.gpu_free
        fits = np.flatnonzero(r >= d - TOL)
        return int(fits[np.argmin(r[fits])])


class FirstFitPolicy:
    """First feasible node in list order; first qualifying GPU for fractional."""

    def decide(self, cluster: Cluster, cls):
        for node in cluster.nodes:
            if node.fits(cls):
                d = cls.gpu_demand
                if 0.0 < d < 1.0:
                    return (node, int(np.flatnonzero(node.gpu_free >= d - TOL)[0]))
                return (node, None)
        return (None, None)


class RandomPolicy:
    """Uniform among feasible nodes; uniform qualifying GPU (paper Baselines 5)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def decide(self, cluster: Cluster, cls):
        feas = cluster.feasible_nodes(cls)
        if not feas:
            return (None, None)
        node = feas[self.rng.integers(len(feas))]
        d = cls.gpu_demand
        if 0.0 < d < 1.0:
            fits = np.flatnonzero(node.gpu_free >= d - TOL)
            return (node, int(fits[self.rng.integers(len(fits))]))
        return (node, None)


class DotProdPolicy:
    """Smallest dot-product of node's normalized remaining resources and task demands.

    Vectors: [cpu_frac, gpu_frac(scalarized)], normalized by max node capacity
    (paper extension 3); demand vector in the same scale.
    """

    def __init__(self, max_cpu_milli: float, max_gpu: float):
        self.max_cpu = float(max_cpu_milli)
        self.max_gpu = float(max_gpu)

    @classmethod
    def from_cluster(cls, cluster: Cluster):
        return cls(max(n.cpu_milli for n in cluster.nodes), max(n.num_gpus for n in cluster.nodes))

    def decide(self, cluster: Cluster, cls):
        best = None
        best_key = None
        cpu_d = cls.cpu_milli / self.max_cpu
        gpu_d = cls.gpu_demand / self.max_gpu
        for node in cluster.nodes:
            if not node.fits(cls):
                continue
            remaining = np.array([node.cpu_free / self.max_cpu, _scalar_gpu_capacity(node) / self.max_gpu])
            key = float(remaining @ np.array([cpu_d, gpu_d]))
            if best_key is None or key < best_key:
                best_key, best = key, node
        if best is None:
            return (None, None)
        return (best, BestFitPolicy._gpu_in_node(self, best, cls))


class PackingPolicy:
    """Tiered placement (paper Baselines 3): occupied shared GPUs first, then
    idle GPUs on nodes with any occupancy, then fully idle nodes. Within a
    tier: least remaining qualifying GPU / most occupied node."""

    def decide(self, cluster: Cluster, cls):
        d = cls.gpu_demand
        candidates = [n for n in cluster.nodes if n.fits(cls)]
        if not candidates:
            return (None, None)
        r_used = np.array([n.used_slots for n in candidates])
        node_rank = np.argsort(-r_used, kind="stable")
        best_node = None
        for node in [candidates[i] for i in node_rank]:
            if d > 0.0 and d < 1.0:
                shared = np.flatnonzero((node.gpu_free > TOL) & (node.gpu_free < 1.0 - TOL))
                usable = [g for g in shared if node.gpu_free[g] >= d - TOL]
                if usable:
                    g = usable[int(np.argmin(node.gpu_free[usable]))]
                    return (node, int(g))
                fits = np.flatnonzero(node.gpu_free >= d - TOL)
                if len(fits):
                    g = fits[int(np.argmin(node.gpu_free[fits]))]
                    best_node = (node, int(g))
                    break
            else:
                best_node = (node, None)
                break
        return best_node if best_node else (None, None)


class ClusteringPolicy:
    """Prefer nodes already hosting tasks of the same GPU demand (paper
    Baselines 4); tie-break BestFit. Needs placement observations
    (PlacementObserver protocol — `observe_placement`, see policies/base)."""

    def __init__(self, max_cpu_milli: float, max_gpu: float):
        self.max_cpu = float(max_cpu_milli)
        self.max_gpu = float(max_gpu)
        self.hosted: dict[str, dict[float, int]] = {}

    @classmethod
    def from_cluster(cls, cluster: Cluster):
        return cls(max(n.cpu_milli for n in cluster.nodes), max(n.num_gpus for n in cluster.nodes))

    def observe_placement(self, node_id: str, cls, placed: bool) -> None:
        hosted = self.hosted.setdefault(node_id, {})
        if placed:
            hosted[cls.gpu_demand] = hosted.get(cls.gpu_demand, 0) + 1
        else:
            left = hosted.get(cls.gpu_demand, 0) - 1
            if left > 0:
                hosted[cls.gpu_demand] = left
            else:
                hosted.pop(cls.gpu_demand, None)

    def decide(self, cluster: Cluster, cls):
        best = None
        best_key = None
        bf = BestFitPolicy(self.max_cpu, self.max_gpu)
        for node in cluster.nodes:
            if not node.fits(cls):
                continue
            same = self.hosted.get(node.id, {}).get(cls.gpu_demand, 0)
            key = (-same, bf._residual(node, cls))
            if best_key is None or key < best_key:
                best_key, best = key, node
        if best is None:
            return (None, None)
        return (best, bf._gpu_in_node(best, cls))
