"""Gang commit machinery (plan/commit split, AGENTS.md §12.4).

Policies never mutate the live cluster: they return pure-data GangPlans and
the runner commits via the two-phase commit here. Kept verbatim — this is
one of the already-strong surfaces the redesign preserves.
"""

FAILURE_REASONS = (
    "no_single_node",
    "insufficient_gpu",
    "insufficient_cpu",
    "insufficient_memory",
    "capacity_fragmented",
    "policy_miss",
    "commit_failed",
)


def _commit_gang_plan(cluster, worker_cls, plan):
    """Two-phase commit: dry-run the whole plan on copies of the INVOLVED nodes
    only (aggregate validation before any real mutation; O(k) not O(N)), then
    apply. On unexpected mid-apply failure, roll back the receipts in reverse —
    never leave half-placed state. Returns the applied [(node, placement)]
    receipts, or None."""
    plan = tuple(plan)
    if not plan:
        return None
    dry_nodes = {}
    try:
        for step in plan:
            dry = dry_nodes.get(step.node_id)
            if dry is None:
                dry = cluster.node_by_id(step.node_id).copy()
                dry_nodes[step.node_id] = dry
            dry.place(worker_cls, gpu_idx=step.gpu_idx)
    except (KeyError, ValueError):
        return None
    applied = []
    try:
        for step in plan:
            node = cluster.node_by_id(step.node_id)
            applied.append((node, node.place(worker_cls, gpu_idx=step.gpu_idx)))
    except (KeyError, ValueError):
        for node, placement in reversed(applied):
            node.release(placement, worker_cls)
        return None
    return applied


def commit_gang_plan(cluster, worker_cls, plan) -> bool:
    """Layer-1 wrapper: success iff the plan committed (receipts discarded)."""
    return _commit_gang_plan(cluster, worker_cls, plan) is not None


def failure_reason(cluster, worker_cls, k: int, packed: bool = False) -> str:
    """Why a gang/item was not placed (enum in FAILURE_REASONS)."""
    if worker_cls.gpu_demand < 1.0:
        if int(k) == 1 or packed:
            return (
                "policy_miss"
                if any(n.fits(worker_cls) for n in cluster.nodes)
                else "insufficient_gpu"
            )
        # fractional gang: total per-node fractional capacity decides
        if worker_cls.gpu_demand <= 0.0:
            return "policy_miss"
        if sum(n.max_frac_workers(worker_cls) for n in cluster.nodes) >= int(k):
            return "policy_miss"
        d = float(worker_cls.gpu_demand)
        gpu_cap = sum(int(((n.gpu_free + 1e-9) // d).sum()) for n in cluster.nodes)
        return "insufficient_gpu" if gpu_cap < int(k) else "capacity_fragmented"
    k = int(k)
    caps = [n.max_whole_workers(worker_cls) for n in cluster.nodes]
    if packed:
        return "no_single_node" if max(caps, default=0) < k else "policy_miss"
    if sum(caps) >= k:
        return "policy_miss"
    d = int(worker_cls.gpu_demand)
    gpu_cap = sum(n.free_slots // d for n in cluster.nodes)
    cpu_cap = sum(
        (n.cpu_free // worker_cls.cpu_milli) if worker_cls.cpu_milli > 0 else k
        for n in cluster.nodes
    )
    mem_cap = sum(
        (n.memory_free // worker_cls.memory_mib) if worker_cls.memory_mib > 0 else k
        for n in cluster.nodes
    )
    if gpu_cap < k:
        return "insufficient_gpu"
    if cpu_cap < k:
        return "insufficient_cpu"
    if mem_cap < k:
        return "insufficient_memory"
    return "capacity_fragmented"
