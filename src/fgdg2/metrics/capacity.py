"""Cluster free-capacity counters (was sim/metrics.py)."""


def unallocated_gpu_shares(cluster) -> float:
    """Sum of free GPU shares over all devices, in GPU units.
    Fully busy GPU → 0, half-used → 0.5, fully free → 1."""
    return float(sum(n.gpu_free.sum() for n in cluster.nodes))


def fully_free_gpus(cluster) -> int:
    return int(sum(n.free_slots for n in cluster.nodes))
