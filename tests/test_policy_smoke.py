import numpy as np
import pandas as pd

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.policies.fgd import FGDPolicy
from fgdg2.sim.engine import TraceEngine


def test_fgd_prefers_node_that_cuts_fragmentation():
    workload = Workload(
        [TaskClass("f5", gpu_demand=0.5), TaskClass("w1", gpu_demand=1, whole_gpu=True)],
        np.array([0.5, 0.5]),
    )
    nodes = [
        Node.with_free("a", [1.0, 0.2], cpu_milli=32000),
        Node.with_free("b", [0.9, 0.9], cpu_milli=32000),
    ]
    task = TaskClass("foo", cpu_milli=1000, gpu_demand=0.5)
    node, g = FGDPolicy(workload).decide(Cluster(nodes), task)
    assert (node.id, g) == ("b", 0)


def test_fgd_tie_breaks_in_node_order():
    workload = Workload([TaskClass("f4", gpu_demand=0.4)], np.array([1.0]))
    nodes = [Node.with_free("a", [0.8, 0.0], cpu_milli=32000), Node.with_free("b", [1.0, 1.0], cpu_milli=32000)]
    task = TaskClass("foo", cpu_milli=1000, gpu_demand=0.4)
    node, g = FGDPolicy(workload).decide(Cluster(nodes), task)
    assert (node.id, g) == ("a", 0)


def test_fgd_keeps_room_for_big_whole_gpu_demand():
    workload = Workload(
        [TaskClass("w2", gpu_demand=2, whole_gpu=True), TaskClass("w1", gpu_demand=1, whole_gpu=True)],
        np.array([0.9, 0.1]),
    )
    nodes = [
        Node.with_free("a", [1.0, 1.0], cpu_milli=32000),
        Node.with_free("b", [1.0], cpu_milli=32000),
    ]
    task = TaskClass("foo", cpu_milli=1000, gpu_demand=1, whole_gpu=True)
    node, _ = FGDPolicy(workload).decide(Cluster(nodes), task)
    assert node.id == "b"


def test_engine_roundtrip_and_unplaced():
    workload = Workload([TaskClass("w1", gpu_demand=1, whole_gpu=True)], np.array([1.0]))
    cluster = Cluster([Node.with_free("a", [1.0], cpu_milli=32000, memory_mib=10000)])
    engine = TraceEngine(cluster, FGDPolicy(workload))
    pods = pd.DataFrame([
        {"name": "p1", "creation_time": 0, "deletion_time": 10, "cpu_milli": 1000,
         "memory_mib": 100, "gpu_demand": 1.0},
        {"name": "p2", "creation_time": 1, "deletion_time": 10, "cpu_milli": 1000,
         "memory_mib": 100, "gpu_demand": 1.0},
        {"name": "p3", "creation_time": 2, "deletion_time": 12, "cpu_milli": 1000,
         "memory_mib": 100, "gpu_demand": 1.0},
    ])
    res = engine.run(pods)
    assert res.placed + res.unplaced == res.arrivals == 3
    assert res.placed == 1 and res.unplaced == 2
    assert res.releases == 1
    assert cluster.nodes[0].free_slots == 1  # flush() released p1 at end of run
    p3 = pd.Series(pods.iloc[2]).to_dict()
    p3["creation_time"] = 13
    p3["name"] = "p4"
    res2 = engine.step(pd.Series(p3))
    assert res2.placed == 1 and res2.releases == 0  # active list empty after flush
    assert cluster.nodes[0].free_slots == 0
