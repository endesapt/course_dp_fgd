"""Fractional gang workers (0 < gpu_demand < 1): DP, seq_fgd, capacity,
routing and Layer-1 accounting.

The fractional DP is exact over the node dimension GIVEN the within-node
greedy FGD rule (`gang_deltas_frac`); these tests check exactly that claim by
brute force, plus that the per-node costs are real fragmentation deltas.
"""


import numpy as np

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.objective import workload_fragmentation
from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.items import StreamItem
from fgdg2.policies.baselines import BestFitPolicy
from fgdg2.policies.gang import (
    AtomicSequentialRawFGDGang,
    DPFGDGang,
    MixedPolicy,
    SequentialGangPolicy,
    gang_deltas_frac,
)
from fgdg2.sim.commit import commit_gang_plan
from fgdg2.sim.gang import GangRunner

FRAC_DEMANDS = (0.25, 0.3, 0.5)


def make_workload(rng):
    classes = []
    for i, d in enumerate((0.25, 0.5, 1.0, 2.0)):
        cpu = int(rng.choice([1000, 4000, 8000]))
        classes.append(TaskClass(f"c{i}", cpu_milli=cpu, gpu_demand=float(d), whole_gpu=d >= 1.0))
    pops = rng.random(len(classes)) + 0.1
    return Workload(tuple(classes), pops / pops.sum())


def make_node(rng, i):
    g = int(rng.integers(1, 5))
    shares = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=g)
    return Node.with_free(f"n{i}", shares, cpu_milli=int(rng.choice([8000, 16000, 32000])))


def compositions(caps, k):
    if len(caps) == 1:
        if 0 <= k <= caps[0]:
            yield (k,)
        return
    for x in range(min(caps[0], k) + 1):
        for rest in compositions(caps[1:], k - x):
            yield (x,) + rest


def alloc_of(cluster, plan):
    ids = [n.id for n in cluster.nodes]
    out = [0] * len(ids)
    for s in plan:
        out[ids.index(s.node_id)] += 1
    return tuple(out)


def test_frac_capacity_matches_greedy_fill():
    rng = np.random.default_rng(1)
    for _ in range(200):
        node = make_node(rng, 0)
        d = float(rng.choice(FRAC_DEMANDS))
        cls = TaskClass("w", cpu_milli=int(rng.choice([0, 1000, 3000])), gpu_demand=d, whole_gpu=False)
        cap = node.max_frac_workers(cls)
        dry = node.copy()
        placed = 0
        while dry.fits(cls):
            g = int(np.flatnonzero(dry.gpu_free >= d - 1e-9)[0])
            dry.place(cls, gpu_idx=g)
            placed += 1
        assert placed == cap


def test_frac_deltas_are_real_fragmentation_deltas():
    rng = np.random.default_rng(2)
    for _ in range(60):
        wl = make_workload(rng)
        cluster = Cluster([make_node(rng, i) for i in range(3)])
        d = float(rng.choice(FRAC_DEMANDS))
        cls = TaskClass("w", cpu_milli=1000, gpu_demand=d, whole_gpu=False)
        caps = np.array([n.max_frac_workers(cls) for n in cluster.nodes])
        kmax = int(caps.max()) if caps.max() > 0 else 1
        delta, seq = gang_deltas_frac(cluster, wl, cls, kmax, caps)
        for i, node in enumerate(cluster.nodes):
            f0 = workload_fragmentation([node], wl)
            dry = node.copy()
            for x in range(1, caps[i] + 1):
                dry.place(cls, gpu_idx=int(seq[i, x - 1]))
                assert abs((workload_fragmentation([dry], wl) - f0) - delta[i, x]) < 1e-6
            assert np.all(np.isinf(delta[i, caps[i] + 1:]))


def test_frac_dp_equals_brute_force_given_within_node_rule():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(150):
        wl = make_workload(rng)
        cluster = Cluster([make_node(rng, i) for i in range(int(rng.integers(2, 4)))])
        d = float(rng.choice(FRAC_DEMANDS))
        cls = TaskClass("w", cpu_milli=int(rng.choice([0, 1000, 4000])), gpu_demand=d, whole_gpu=False)
        k = int(rng.integers(1, 6))
        caps = [n.max_frac_workers(cls) for n in cluster.nodes]
        plan = DPFGDGang(wl).decide_gang(cluster, cls, k)
        if sum(caps) < k:
            assert plan is None
            continue
        delta, _ = gang_deltas_frac(cluster, wl, cls, k, np.array(caps))
        best = min(sum(delta[i, a] for i, a in enumerate(al)) for al in compositions(caps, k))
        assert plan is not None and len(plan) == k
        got = sum(delta[i, a] for i, a in enumerate(alloc_of(cluster, plan)))
        assert abs(got - best) < 1e-9
        dry = cluster.copy()
        assert commit_gang_plan(dry, cls, plan)
        checked += 1
    assert checked > 50


def test_frac_seq_equals_dp_at_k1_and_commits():
    rng = np.random.default_rng(4)
    for _ in range(100):
        wl = make_workload(rng)
        cluster = Cluster([make_node(rng, i) for i in range(3)])
        d = float(rng.choice(FRAC_DEMANDS))
        cls = TaskClass("w", cpu_milli=1000, gpu_demand=d, whole_gpu=False)
        p_dp = DPFGDGang(wl).decide_gang(cluster, cls, 1)
        p_seq = AtomicSequentialRawFGDGang(wl).decide_gang(cluster, cls, 1)
        assert p_dp == p_seq
        for k in (2, 4):
            p = AtomicSequentialRawFGDGang(wl).decide_gang(cluster, cls, k)
            if p is not None:
                assert len(p) == k and commit_gang_plan(cluster.copy(), cls, p)


def test_whole_dp_unchanged_by_x_range_restriction():
    """Restricting DP candidates to x <= max cap must not change whole plans."""
    rng = np.random.default_rng(5)
    for _ in range(80):
        classes = [TaskClass(f"w{i}", cpu_milli=1000 * (i + 1), gpu_demand=float(i + 1), whole_gpu=True)
                   for i in range(3)]
        wl = Workload(tuple(classes), np.array([0.5, 0.3, 0.2]))
        nodes = [Node.with_free(f"n{i}", rng.choice([0.0, 1.0], size=int(rng.integers(1, 9))),
                                cpu_milli=32000) for i in range(4)]
        cluster = Cluster(nodes)
        cls = classes[int(rng.integers(0, 2))]
        k = int(rng.integers(1, 7))
        caps = [n.max_whole_workers(cls) for n in nodes]
        plan = DPFGDGang(wl).decide_gang(cluster, cls, k)
        if sum(caps) < k:
            assert plan is None
            continue
        # exhaustive objective check (same as test_gang_dp, independent of X)
        f0 = workload_fragmentation(cluster.nodes, wl)
        best = None
        for al in compositions(caps, k):
            dry = cluster.copy()
            ok = True
            for node, a in zip(dry.nodes, al):
                for _ in range(a):
                    if not node.fits(cls):
                        ok = False
                        break
                    node.place(cls)
            if ok:
                v = workload_fragmentation(dry.nodes, wl) - f0
                best = v if best is None else min(best, v)
        dry = cluster.copy()
        assert commit_gang_plan(dry, cls, plan)
        assert abs((workload_fragmentation(dry.nodes, wl) - f0) - best) < 1e-6


def test_gang_routing_and_runner_accounting_on_mixed_stream():
    rng = np.random.default_rng(6)
    wl = make_workload(rng)
    nodes = [Node(f"n{i}", 8, cpu_milli=96000) for i in range(4)]
    items = []
    specs = [(6, 0.5), (1, 0.25), (3, 1.0), (10, 0.25), (1, 2.0), (4, 0.5), (2, 0.3)]
    for i, (k, d) in enumerate(specs):
        items.append(StreamItem(gang_id=f"g{i}", gang_size=k, worker_cpu_milli=1000,
                                worker_gpu_demand=d, arrival_index=i, is_gang=k > 1))
    for gang in (DPFGDGang(wl), AtomicSequentialRawFGDGang(wl),
                 SequentialGangPolicy(BestFitPolicy(96000, 8))):
        cluster = Cluster([n.copy() for n in nodes])
        res = GangRunner(cluster, MixedPolicy(gang, None), capacity=32.0).run(items)
        # runner raises AssertionError if incremental counters diverge
        assert res.gangs_total == sum(1 for k, _ in specs if k > 1)
        assert res.items_placed == len(specs)
