"""FGD-exact (Weng et al. ATC'23 Alg. 1) — naive + batched decision paths.

Verbatim port: argmin DeltaF_n(M) over feasible (node, gpu); quantized=True
reproduces the reference scorer (sigmoid quantization, int64 truncation).
"""

import numpy as np

from ..core.cluster import Cluster
from ..core.node import Node
from ..core.objective import (
    node_workload_fragmentation,
    raw_node_workload_fragmentation,
    raw_node_workload_fragmentation_batch,
)
from ..core.rmath import TOL


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def quantized_score(delta: float) -> int:
    """Reference-FGD score: improvement = -delta in GPU units (1 unit = 1000
    milli); score = int64(100 * sigmoid(improvement)) with int truncation.
    Improvements above ~4.6 GPU units saturate and tie."""
    improvement = -delta
    return int(_sigmoid(improvement) * 100)


def naive_decide(nodes, workload, cls, quantized: bool = False):
    """Reference implementation: O(N·G) explicit loop over all candidates.

    All pods (including CPU-only) go through the same DeltaF scoring, as in the
    reference FGD: a CPU-only placement is scored by the CPU-induced flips of
    other classes' Case-1/Case-2 status. Ties keep the first node in list order;
    the reference resolves ties lexicographically on per-seed randomly prefixed
    node names (i.e. an effectively random node) — a documented deviation, §9.6.
    """
    d = cls.gpu_demand
    best: tuple[float, Node, int | None] | None = None
    for node in nodes:
        if not node.fits(cls):
            continue
        f_before = node_workload_fragmentation(node, workload)
        cpu_after = node.cpu_free - cls.cpu_milli
        if d == 0.0:
            delta = raw_node_workload_fragmentation(node.gpu_free, workload, cpu_after) - f_before
            key = quantized_score(delta) if quantized else -delta
            if best is None or key > best[0]:
                best = (key, node, None)
            continue
        gpu_candidates: list[int | None] = (
            [g for g in range(node.num_gpus) if node.gpu_free[g] >= d - TOL] if d < 1.0 else [None]
        )
        for g in gpu_candidates:
            r_after = node.gpu_free.copy()
            if g is not None:
                r_after[g] -= d
            else:
                full = np.flatnonzero(r_after >= 1.0 - TOL)
                r_after[full[: int(d)]] = 0.0
            delta = raw_node_workload_fragmentation(r_after, workload, cpu_after) - f_before
            key = quantized_score(delta) if quantized else -delta
            if best is None or key > best[0]:
                best = (key, node, g)
    return (best[1], best[2]) if best else (None, None)


class FGDPolicy:
    """FGD-exact, Weng et al. Alg. 1: argmin DeltaF_n(M) over feasible (node, gpu).

    quantized=True reproduces the reference scorer (sigmoid quantization with
    int64 truncation and lexicographic tie-breaks) instead of exact argmin.
    """

    def __init__(self, workload, quantized: bool = False):
        self.workload = workload
        self.quantized = quantized

    def decide(self, cluster: Cluster, cls):
        return naive_decide(cluster.nodes, self.workload, cls, quantized=self.quantized)


def fast_decide(nodes, workload, cls, quantized: bool = False):
    """Batched equivalent of naive_decide: identical candidate set, objective
    values and tie-breaking, vectorized over nodes and GPU columns."""
    d = cls.gpu_demand
    n = len(nodes)
    if n == 0:
        return (None, None)
    gmax = max(nd.num_gpus for nd in nodes)
    R = np.full((n, gmax), np.nan)
    for i, nd in enumerate(nodes):
        R[i, : nd.num_gpus] = nd.gpu_free
    valid = ~np.isnan(R)
    cpu_free = np.array([nd.cpu_free for nd in nodes])
    fits_base = np.array([nd.fits_non_gpu(cls) for nd in nodes])
    F_before = raw_node_workload_fragmentation_batch(R, valid, cpu_free, workload)
    cpu_after = cpu_free - cls.cpu_milli

    def score(delta):
        if not quantized:
            return -delta
        improvement = -np.asarray(delta)
        return np.trunc(100.0 / (1.0 + np.exp(-improvement))).astype(np.int64)

    if d == 0.0:
        key = score(raw_node_workload_fragmentation_batch(R, valid, cpu_after, workload) - F_before)
        key = np.where(fits_base, key, -np.inf)
        i = int(np.argmax(key))
        return (nodes[i], None) if np.isfinite(key[i]) else (None, None)

    if d >= 1.0:
        full_rank = np.cumsum(valid & (R >= 1.0 - TOL), axis=1)
        selected = valid & (R >= 1.0 - TOL) & (full_rank <= int(d))
        fits_gpu = (valid & (R >= 1.0 - TOL)).sum(axis=1) >= int(d)
        R_after = np.where(selected, 0.0, R)
        F_after = raw_node_workload_fragmentation_batch(R_after, valid, cpu_after, workload)
        key = score(F_after - F_before)
        key = np.where(fits_base & fits_gpu, key, -np.inf)
        i = int(np.argmax(key))
        return (nodes[i], None) if np.isfinite(key[i]) else (None, None)

    # fractional: one column at a time, candidates r_g >= d
    keys = np.full((n, gmax), -np.inf)
    for g in range(gmax):
        cand = valid[:, g] & (R[:, g] >= d - TOL)
        if not cand.any():
            continue
        R_after = R.copy()
        R_after[cand, g] = R[cand, g] - d
        F_after = raw_node_workload_fragmentation_batch(R_after, valid, cpu_after, workload)
        k = score(F_after - F_before)
        keys[:, g] = np.where(cand & fits_base, k, -np.inf)
    if not np.isfinite(keys).any():
        return (None, None)
    flat = int(np.argmax(keys))
    i, g = divmod(flat, gmax)
    return (nodes[i], g)


class FastFGDPolicy:
    """FGD-exact with the batched decision path; oracle = naive_decide."""

    def __init__(self, workload, quantized: bool = False):
        self.workload = workload
        self.quantized = quantized

    def decide(self, cluster: Cluster, cls):
        return fast_decide(cluster.nodes, self.workload, cls, quantized=self.quantized)
