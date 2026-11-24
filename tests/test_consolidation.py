"""Phase-2 elasticity: consolidation-beta knob + releasable-capacity metric.

beta semantics: +beta GPU-shares per node a placement leaves PARTIALLY free
(0 < free_after < num_gpus) — provider-binary objective: full and empty nodes
are releasable-neutral; a partial slab is stranded capacity.
"""

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.tasks import workload_from_counts
from fgdg2.policies.gang import DPFGDGang, FGDIntegerPolicy


def test_node_is_empty():
    a = Node("a", 4, cpu_milli=64000)
    assert a.is_empty
    cls = workload_from_counts({(1000, 1.0): 1}).classes[0]
    a.place(cls)
    assert not a.is_empty


def test_beta_fills_partial_node_dp():
    """Node 'b' has exactly 1 free GPU — a d=1 worker FILLS it (after=0, no
    penalty). Node 'a' is empty — the worker leaves it partial (+beta).
    Pure ΔF ties (single-class M) -> beta=0 picks 'a' (list order), beta>0
    must pick 'b'."""
    wl = workload_from_counts({(1000, 1.0): 1})
    cls = wl.classes[0]
    a = Node("a", 4, cpu_milli=64000)
    b = Node("b", 4, cpu_milli=64000)
    for _ in range(3):  # b: 3 used, 1 free
        b.place(cls)
    cluster = Cluster([a, b])
    p0 = DPFGDGang(wl, beta=0.0).decide_gang(cluster, cls, 1)
    p1 = DPFGDGang(wl, beta=1.0).decide_gang(cluster, cls, 1)
    assert {s.node_id for s in p0} == {"a"}
    assert {s.node_id for s in p1} == {"b"}


def test_beta_fills_partial_node_sequential():
    wl = workload_from_counts({(1000, 1.0): 1})
    cls = wl.classes[0]
    a = Node("a", 4, cpu_milli=64000)
    b = Node("b", 4, cpu_milli=64000)
    for _ in range(3):
        b.place(cls)
    cluster = Cluster([a, b])
    node, _ = FGDIntegerPolicy(wl, beta=1.0).decide(cluster, cls)
    assert node.id == "b"


def test_beta_consolidates_spread_gang():
    """k=2 spread gang on 3 empty nodes: beta>0 should put both workers on ONE
    node (1 partial slab) rather than 1+1 (2 partial slabs)."""
    wl = workload_from_counts({(1000, 1.0): 1})
    nodes = [Node(f"n{i}", 4, cpu_milli=64000) for i in range(3)]
    cluster = Cluster(nodes)
    plan = DPFGDGang(wl, beta=1.0).decide_gang(cluster, wl.classes[0], 2)
    assert plan is not None and len({s.node_id for s in plan}) == 1


def test_beta_filling_empty_node_free():
    """Empty node filled COMPLETELY (x_after=0) pays no beta — a packed
    whole-node job is releasable-neutral."""
    wl = workload_from_counts({(1000, 4.0): 1})
    a = Node("a", 4, cpu_milli=64000)
    cluster = Cluster([a])
    plan = DPFGDGang(wl, beta=10.0).decide_gang(cluster, wl.classes[0], 1)
    assert plan is not None


def test_empty_node_hours_accumulates():
    """Releasable metric: ∫empty_nodes dt must reflect actual empty time."""
    from fgdg2.sim.churn import ChurnRunner

    class _Reject:
        def decide_gang(self, cluster, cls, k):
            return None

    node = Node("n", 8, cpu_milli=64000)
    cluster = Cluster(nodes=[node])
    items = []
    for i in range(3):
        items.append({"worker_cpu_milli": 8000, "worker_gpu_demand": 99.0,
                      "submit_time_s": float(i * 10), "arrival_index": i,
                      "is_gang": False, "gang_size": 1, "gang_id": f"g{i}",
                      "duration_s": 5.0})
    res = ChurnRunner(cluster, _Reject(), capacity=8.0).run(items)
    # node never used -> empty for the whole span (arrivals at t=0,10,20)
    assert res.empty_node_hours == 20.0
    assert res.unalloc_gpu_hours == 20.0 * 8
    assert res.nodes_never_used == 1
    assert res.node_max_empty_streak["n"] == res.span


def test_cluster_add_remove_node():
    """Elastic membership: add/remove keep the id index consistent; duplicate
    ids rejected; removed nodes are gone from lookups and totals."""
    import pytest

    a, b = Node("a", 4, cpu_milli=64000), Node("b", 4, cpu_milli=64000)
    cluster = Cluster([a])
    cluster.add_node(b)
    assert cluster.total_gpus == 8 and cluster.node_by_id("b") is b
    with pytest.raises(ValueError):
        cluster.add_node(Node("a", 4))
    cluster.remove_node("a")
    assert cluster.total_gpus == 4 and len(cluster.nodes) == 1
    with pytest.raises(KeyError):
        cluster.node_by_id("a")


def test_tracker_evict_add_counters():
    """Evict closes the streak and drops the node's free capacity from every
    counter; add restores them; consistency holds throughout."""
    from fgdg2.sim.capacity import CapacityTracker

    cls = workload_from_counts({(1000, 1.0): 1}).classes[0]
    a, b = Node("a", 4, cpu_milli=64000), Node("b", 4, cpu_milli=64000)
    cluster = Cluster([a, b])
    tr = CapacityTracker(cluster)
    assert tr.online_nodes == 2 and tr.empty_nodes == 2
    # occupy 'a' partially — a non-empty node cannot be evicted
    node_a = cluster.node_by_id("a")
    pl = cluster.place(node_a, cls)
    tr.place(cls, [(node_a, pl)], ["a"], ["a"], 0, now=10.0)
    import pytest
    with pytest.raises(ValueError):
        tr.evict("a", now=20.0)
    tr.evict("b", now=20.0)
    assert tr.online_nodes == 1 and tr.online_gpus == 4
    assert tr.empty_nodes == 0 and tr.fully_free == 3
    assert tr.max_empty_streak["b"] == 20.0  # streak closed at evict time
    tr.assert_consistent()
    tr.add(Node("c", 4, cpu_milli=64000), now=30.0)
    assert tr.online_nodes == 2 and tr.online_gpus == 8
    assert tr.empty_since["c"] == 30.0
    tr.assert_consistent()


def test_release_returns_newly_emptied():
    """release() reports exactly the nodes that became fully empty — the
    eviction-check arming signal."""
    from fgdg2.sim.capacity import CapacityTracker

    wl = workload_from_counts({(1000, 1.0): 1})
    cls = wl.classes[0]
    a = Node("a", 4, cpu_milli=64000)
    cluster = Cluster([a])
    tr = CapacityTracker(cluster)
    pls = [(cluster.node_by_id("a"), cluster.place(cluster.node_by_id("a"), cls))
           for _ in range(2)]
    tr.place(cls, pls, ["a"], ["a"], 0, now=0.0)
    # first release: node still has 1 worker -> not empty
    assert tr.release(cls, pls[:1], now=5.0) == []
    # second release: fully empty -> reported
    assert tr.release(cls, pls[1:], now=9.0) == ["a"]


def test_never_used_counts_ever_online():
    """nodes_never_used is over ever-online ids, not the live pool — an
    evicted-and-used node doesn't resurrect as 'never used'."""
    from fgdg2.sim.capacity import CapacityTracker

    a = Node("a", 4, cpu_milli=64000)
    tr = CapacityTracker(Cluster([a]))
    tr.evict("a", now=10.0)
    tr.add(Node("b", 4, cpu_milli=64000), now=20.0)
    assert tr.nodes_never_used == 2  # a and b both online at some point, unused


def test_node_streak_closes_on_use():
    """A node used then released has streak = release→end, not the full span;
    an untouched node keeps a full-span streak."""
    from fgdg2.policies.gang import DPFGDGang
    from fgdg2.sim.churn import ChurnRunner

    wl = workload_from_counts({(1000, 1.0): 1})
    a = Node("a", 4, cpu_milli=64000)
    b = Node("b", 4, cpu_milli=64000)
    cluster = Cluster([a, b])
    # arrivals at t=0 and t=20 (duration 10 each): both land on 'a' (node
    # order tie-break), so 'a' is empty 10->20 as its longest streak;
    # 'b' is never touched.
    items = [{"worker_cpu_milli": 1000, "worker_gpu_demand": 1.0,
              "submit_time_s": float(t), "arrival_index": i,
              "is_gang": False, "gang_size": 1, "gang_id": f"g{i}",
              "duration_s": 10.0} for i, t in enumerate((0.0, 20.0))]
    res = ChurnRunner(cluster, DPFGDGang(wl), capacity=8.0).run(items)
    assert res.nodes_never_used == 1
    assert res.span == 30.0
    # open streaks clip at the last arrival (t=20), not the drain-tail end
    assert res.node_max_empty_streak["b"] == 20.0
    assert res.node_max_empty_streak["a"] == 10.0
