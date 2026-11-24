"""DP correctness tests (AGENTS.md §12.5): DP == brute force on random small instances.

Brute force enumerates ALL allocation vectors and scores each by DRY-RUNNING the
plan on a cluster copy and measuring the raw F_cluster delta (end-to-end validation
of `gang_deltas`, not just self-consistency).
"""

from collections import Counter

import numpy as np
import pytest

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.objective import workload_fragmentation
from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.policies.gang import DPFGDGang, plan_for_allocation
from fgdg2.sim.commit import commit_gang_plan


def make_workload(rng, n_classes=None):
    n = n_classes or int(rng.integers(2, 4))
    classes = []
    for i in range(n):
        d = int(rng.integers(1, 4))
        cpu = int(rng.choice([1000, 4000, 8000, 16000]))
        classes.append(TaskClass(f"w{i}", cpu_milli=cpu, gpu_demand=float(d), whole_gpu=True))
    pops = rng.random(n) + 0.1
    return Workload(classes, pops / pops.sum())


def compositions(caps, k):
    """All (a_1, ..., a_N) with 0 <= a_i <= caps_i and sum == k."""
    if len(caps) == 1:
        if 0 <= k <= caps[0]:
            yield (k,)
        return
    for x in range(min(caps[0], k) + 1):
        for rest in compositions(caps[1:], k - x):
            yield (x,) + rest


def make_node(rng, i):
    g = int(rng.integers(0, 9))
    shares = rng.choice([0.0, 1.0], size=g)
    if g and rng.random() < 0.25:
        shares[int(rng.integers(g))] = 0.5
    return Node.with_free(f"n{i}", shares, cpu_milli=int(rng.choice([8000, 16000, 32000])))


def plan_delta(cluster, workload, cls, plan):
    f_before = workload_fragmentation(cluster.nodes, workload)
    dry = cluster.copy()
    if not commit_gang_plan(dry, cls, plan):
        return None
    return workload_fragmentation(dry.nodes, workload) - f_before


def test_dp_equals_brute_force_on_random_instances():
    rng = np.random.default_rng(20260916)
    instances = 0
    feasible_instances = 0
    for _ in range(200):
        cluster = Cluster([make_node(rng, i) for i in range(int(rng.integers(1, 7)))])
        workload = make_workload(rng)
        cpu = int(rng.choice([1000, 4000, 8000]))
        worker = TaskClass(
            "worker", cpu_milli=cpu, gpu_demand=float(rng.integers(1, 3)), whole_gpu=True
        )
        # reach CPU-boundary states (AGENTS §9.3.7 lesson): cpu_free == class cpu
        if rng.random() < 0.5 and cluster.nodes:
            nd = cluster.nodes[int(rng.integers(len(cluster.nodes)))]
            nd._used_cpu = max(0, nd.cpu_milli - int(rng.integers(0, 3)) * cpu)
        k = int(rng.integers(1, 7))
        caps = [nd.max_whole_workers(worker) for nd in cluster.nodes]
        all_allocs = list(compositions(caps, k))
        dp = DPFGDGang(workload)
        plan = dp.decide_gang(cluster, worker, k)
        instances += 1
        if not all_allocs:
            assert plan is None
            continue
        feasible_instances += 1
        best = min(
            plan_delta(cluster, workload, worker, plan_for_allocation(cluster, worker, alloc))
            for alloc in all_allocs
        )
        assert plan is not None
        got = plan_delta(cluster, workload, worker, plan)
        assert got is not None
        assert got == pytest.approx(best, abs=1e-9)
        assert len(plan) == k
        dry = cluster.copy()
        free_before = dry.total_free_slots
        assert commit_gang_plan(dry, worker, plan)
        assert dry.total_free_slots == free_before - k * int(worker.gpu_demand)
    assert instances >= 100, f"only {instances} instances"
    assert feasible_instances >= 80, f"only {feasible_instances} feasible instances"


def test_dp_edges():
    workload = Workload(
        [TaskClass("w", cpu_milli=1000, gpu_demand=1, whole_gpu=True)], np.array([1.0])
    )
    worker = workload.classes[0]
    dp = DPFGDGang(workload)
    assert dp.decide_gang(Cluster([]), worker, 1) is None
    cluster = Cluster([Node("n0", 2, cpu_milli=8000)])
    plan = dp.decide_gang(cluster, worker, 2)
    assert plan is not None and len(plan) == 2
    assert dp.decide_gang(cluster, worker, 3) is None
    with pytest.raises(ValueError):
        dp.decide_gang(cluster, worker, 0)


def test_dp_uses_multiple_nodes_when_beneficial():
    """Spread semantics non-vacuity: a packed-feasible gang may be spread by DP."""
    load = Workload(
        [TaskClass("w", cpu_milli=1000, gpu_demand=1, whole_gpu=True)], np.array([1.0])
    )
    # one 4-GPU node with exactly 1 worker of CPU left vs four free 1-GPU nodes
    cluster = Cluster(
        [
            Node("big", 4, cpu_milli=2000),
            Node("a", 1, cpu_milli=8000),
            Node("b", 1, cpu_milli=8000),
            Node("c", 1, cpu_milli=8000),
        ]
    )
    plan = DPFGDGang(load).decide_gang(cluster, load.classes[0], 3)
    assert plan is not None and len({s.node_id for s in plan}) > 1


def test_dp_with_fractional_classes_matches_brute_force():
    """The objective M may contain fractional background classes (schema v2); the
    DP must still be one-step optimal under the raw F_n(M)."""
    rng = np.random.default_rng(77)
    checked = 0
    for _ in range(40):
        cluster = Cluster([make_node(rng, i) for i in range(int(rng.integers(2, 6)))])
        whole = make_workload(rng)
        mixed = Workload(
            tuple(whole.classes)
            + (
                TaskClass("b25", cpu_milli=int(rng.choice([1000, 4000])), gpu_demand=0.25),
                TaskClass("b50", cpu_milli=int(rng.choice([2000, 8000])), gpu_demand=0.5),
            ),
            np.concatenate([whole.popularities * 0.7, np.array([0.2, 0.1])]),
        )
        worker = whole.classes[int(rng.integers(len(whole.classes)))]
        k = int(rng.integers(1, 5))
        caps = [nd.max_whole_workers(worker) for nd in cluster.nodes]
        allocs = list(compositions(caps, k))
        plan = DPFGDGang(mixed).decide_gang(cluster, worker, k)
        if not allocs:
            assert plan is None
            continue
        checked += 1
        best = min(
            plan_delta(cluster, mixed, worker, plan_for_allocation(cluster, worker, a))
            for a in allocs
        )
        got = plan_delta(cluster, mixed, worker, plan)
        assert got is not None and got == pytest.approx(best, abs=1e-9)
    assert checked >= 20


def test_dp_heterogeneous_textbook_example():
    """The worked example from the coursework: free GPU 4/3/4, k=4, d=2.

    The potential from Table 2.1 makes node 2 locally attractive for the
    first worker, but the DP should keep it free and place (2,0,2).
    """
    workload = Workload(
        [
            TaskClass("k1", cpu_milli=1000, gpu_demand=1.0, whole_gpu=True),
            TaskClass("k2", cpu_milli=1000, gpu_demand=2.0, whole_gpu=True),
            TaskClass("k4", cpu_milli=1000, gpu_demand=4.0, whole_gpu=True),
        ],
        np.array([0.5, 0.3, 0.2]),
    )
    cluster = Cluster([
        Node.with_free("n1", np.ones(4), cpu_milli=32000),
        Node.with_free("n2", np.ones(3), cpu_milli=32000),
        Node.with_free("n3", np.ones(4), cpu_milli=32000),
    ])
    worker = TaskClass("worker", cpu_milli=1000, gpu_demand=2.0, whole_gpu=True)
    plan = DPFGDGang(workload).decide_gang(cluster, worker, 4)
    assert plan is not None
    assert len(plan) == 4
    counts = Counter(s.node_id for s in plan)
    assert counts == {"n1": 2, "n3": 2}
