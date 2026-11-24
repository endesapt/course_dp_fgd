import numpy as np

from .node import Node
from .rmath import TOL
from .tasks import TaskClass, Workload

FRAG_ROUND = 9  # decimal places; suppresses 1e-15 reduction-order noise so scalar and
# batched implementations agree on a 1e-9 grid (empirically verified, not bit-wise).


def node_fragmentation(node: Node, cls: TaskClass) -> float:
    """Task-level GPU fragmentation F_n(m), Weng et al. Eqs. 2-3.

    Feasibility (Case-1/2 branch) routes through Node.fits — the single home
    of constraints (CPU, model, whole/fractional GPU semantics).
    """
    r = node.gpu_free
    total = float(r.sum())
    if not node.fits(cls) or cls.gpu_demand == 0.0:
        return total
    threshold = min(cls.gpu_demand, 1.0)
    return float(r[r < threshold].sum())


def raw_node_workload_fragmentation(
    r: np.ndarray, workload: Workload, cpu_free: int | None = None
) -> float:
    """F_n(M) computed from free shares + free CPU, Weng et al. Eqs. 1-3.

    Case 1 fires when the node cannot host class m by GPU or by CPU
    (cpu_free is required to evaluate that; None = no CPU constraint,
    for backward-compatible use in tests).
    """
    demands = workload.demand_array
    pops = workload.popularities
    if len(r) == 0 or len(demands) == 0:
        return 0.0
    total = float(r.sum())
    full = int(np.count_nonzero(r >= 1.0 - TOL))
    sorted_r = np.sort(r)
    prefix = np.concatenate([[0.0], np.cumsum(sorted_r)])
    thresholds = np.minimum(demands, 1.0)
    counts = np.searchsorted(sorted_r, thresholds, side="left")
    case2 = prefix[counts]
    fits = np.empty(len(demands), dtype=bool)
    frac = demands < 1.0
    fits[frac] = r.max() + TOL >= demands[frac]
    fits[~frac] = full >= demands[~frac].astype(int)
    if cpu_free is not None:
        cpu_ok = workload.cpu_array <= cpu_free
        fits &= cpu_ok
        fits &= demands > TOL  # d=0 classes stay Case 3 regardless of CPU
    fn = np.where(fits, case2, total)
    fn = np.where(demands <= TOL, total, fn)
    return float(np.round(np.dot(pops, fn), FRAG_ROUND))


def raw_node_workload_fragmentation_batch(
    R: np.ndarray, valid: np.ndarray, cpu_free: np.ndarray, workload: Workload
) -> np.ndarray:
    """Batched F_n(M) for all nodes: R (N, G) free shares, valid (N, G) mask,
    cpu_free (N,). Mirrors raw_node_workload_fragmentation per row."""
    demands = workload.demand_array
    pops = workload.popularities
    n, gmax = R.shape
    if n == 0 or len(demands) == 0:
        return np.zeros(n)
    if gmax == 0:
        return np.zeros(n)  # GPU-less cluster: no GPU can be fragmented
    total = np.where(valid, R, 0.0).sum(axis=1)
    full = (valid & (R >= 1.0 - TOL)).sum(axis=1)
    sorted_inf = np.sort(np.where(valid, R, np.inf), axis=1)
    contrib = np.where(np.isinf(sorted_inf), 0.0, sorted_inf)
    prefix = np.concatenate([np.zeros((n, 1)), np.cumsum(contrib, axis=1)], axis=1)
    thresholds = np.minimum(demands, 1.0)
    counts = (sorted_inf[:, None, :] < thresholds[None, :, None]).sum(axis=2)
    case2 = prefix[np.arange(n)[:, None], counts]
    fits = np.empty((n, len(demands)), dtype=bool)
    frac = demands < 1.0
    rmax = np.where(valid, R, -np.inf).max(axis=1)
    fits[:, frac] = (rmax[:, None] + TOL) >= thresholds[None, frac]
    fits[:, ~frac] = full[:, None] >= demands[None, ~frac].astype(int)
    fits &= cpu_free[:, None] >= workload.cpu_array[None, :]
    fits |= demands[None, :] <= TOL
    fn = np.where(fits, case2, total[:, None])
    fn = np.where(demands[None, :] <= TOL, total[:, None], fn)
    return np.round(fn @ pops, FRAG_ROUND)


def node_workload_fragmentation(node: Node, workload: Workload) -> float:
    """F_n(M) at a node, Eq. 1 (CPU-aware)."""
    return raw_node_workload_fragmentation(node.gpu_free, workload, node.cpu_free)


def workload_fragmentation(nodes, workload: Workload) -> float:
    """F_cluster(M), Eq. 5."""
    return float(sum(node_workload_fragmentation(n, workload) for n in nodes))


def fragmentation_rate(nodes, workload: Workload) -> float:
    """Rate-normalized, Eqs. 4/6: F_cluster over total free GPU capacity."""
    total_free = float(sum(n.total_free_gpu_shares() for n in nodes))
    if total_free <= 0:
        return 0.0
    return workload_fragmentation(nodes, workload) / total_free
