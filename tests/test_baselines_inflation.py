import numpy as np
import pandas as pd

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.tasks import TaskClass
from fgdg2.policies.baselines import (
    BestFitPolicy,
    ClusteringPolicy,
    DotProdPolicy,
    FirstFitPolicy,
    PackingPolicy,
    RandomPolicy,
)
from fgdg2.sim.inflation import InflationRunner


def make_cluster() -> Cluster:
    return Cluster([
        Node.with_free("a", [1.0, 1.0], cpu_milli=32000),
        Node.with_free("b", [0.5, 0.5], cpu_milli=32000),
        Node.with_free("c", [0.8, 0.1], cpu_milli=32000),
    ])


def test_bestfit_picks_least_remaining_node():
    cluster = make_cluster()
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    node, g = BestFitPolicy.from_cluster(cluster).decide(cluster, task)
    assert (node.id, g) == ("c", 0)
    task_w = TaskClass("w1", cpu_milli=1000, gpu_demand=1, whole_gpu=True)
    node, _ = BestFitPolicy.from_cluster(cluster).decide(cluster, task_w)
    assert node.id == "a"


def test_firstfit_picks_first_feasible():
    cluster = make_cluster()
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    node, g = FirstFitPolicy().decide(cluster, task)
    assert node.id == "a" and g == 0


def test_random_is_seeded():
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    r1 = RandomPolicy(np.random.default_rng(3)).decide(make_cluster(), task)
    r2 = RandomPolicy(np.random.default_rng(3)).decide(make_cluster(), task)
    assert r1 == r2


def test_packing_prefers_occupied_gpu():
    cluster = make_cluster()
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    node, g = PackingPolicy().decide(cluster, task)
    assert (node.id, g) == ("b", 1) or (node.id, g) == ("b", 0)


def test_clustering_groups_same_demand():
    cluster = make_cluster()
    policy = ClusteringPolicy.from_cluster(cluster)
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    policy.observe_placement("c", task, True)
    policy.observe_placement("c", task, True)
    node, _ = policy.decide(cluster, task)
    assert node.id == "c"


def test_dotprod_picks_tightest_shape():
    cluster = make_cluster()
    task = TaskClass("f5", cpu_milli=1000, gpu_demand=0.5)
    node, _ = DotProdPolicy.from_cluster(cluster).decide(cluster, task)
    assert node.id == "b"


def test_inflation_crosses_and_measures():
    cluster = make_cluster()
    pods = pd.DataFrame([
        {"name": f"p{i}", "creation_time": 0, "deletion_time": 10, "cpu_milli": 1000,
         "memory_mib": 100, "gpu_demand": 1.0}
        for i in range(10)
    ])
    runner = InflationRunner(cluster, BestFitPolicy.from_cluster(cluster), capacity=4.0, stop_fraction=1.0)
    res = runner.run(pods, seed=0)
    assert res.crossed
    assert res.placed + res.unplaced == len(res.timeline)
    assert res.cumulative_demand >= 4.0
    assert res.unallocated_at_crossing is not None and 0.0 <= res.unallocated_at_crossing <= 4.0
    assert res.fully_free_at_crossing is not None
    assert res.decide_runtime_summary["p50"] >= 0.0
