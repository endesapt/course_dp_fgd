import numpy as np
import pytest
from conftest import NODES_CSV, PODS_CSV, needs_trace

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.policies.fgd import FastFGDPolicy, FGDPolicy, fast_decide, naive_decide

SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEMANDS = [0.0, 0.25, 0.5, 1.0, 2.0]


def make_workload(rng):
    demands = rng.choice([0.0, 0.25, 0.4, 0.5, 1.0, 2.0, 4.0], size=6)
    classes = [
        TaskClass(f"c{i}", cpu_milli=int(rng.integers(0, 4000)), gpu_demand=float(d),
                  whole_gpu=(d >= 1.0))
        for i, d in enumerate(demands)
    ]
    pops = rng.random(len(classes))
    return Workload(classes, pops / pops.sum())


def make_cluster(rng, n_nodes=12):
    nodes = []
    for i in range(n_nodes):
        g = int(rng.integers(0, 9))
        shares = rng.choice(SHARES, size=g)
        node = Node.with_free(
            f"n{i:03d}", shares, model="T",
            cpu_milli=32000, memory_mib=10**6,
        )
        used_cpu = int(rng.integers(0, 30000))
        used_mem = int(rng.integers(0, 10**6))
        node._used_cpu = used_cpu
        node._used_memory = used_mem
        nodes.append(node)
    return Cluster(nodes)


def test_fast_equals_naive_exact_and_quantized():
    rng = np.random.default_rng(42)
    workload = make_workload(rng)
    mismatches = 0
    for trial in range(400):
        cluster = make_cluster(rng)
        d = float(rng.choice(DEMANDS))
        cls = TaskClass(
            f"t{trial}",
            cpu_milli=int(rng.integers(0, 5000)),
            memory_mib=int(rng.integers(0, 10**5)),
            gpu_demand=d,
            whole_gpu=(d >= 1.0),
        )
        for quantized in (False, True):
            a_node, a_g = naive_decide(cluster.nodes, workload, cls, quantized=quantized)
            b_node, b_g = fast_decide(cluster.nodes, workload, cls, quantized=quantized)
            ka = (a_node.id, a_g) if a_node is not None else None
            kb = (b_node.id, b_g) if b_node is not None else None
            if ka != kb:
                mismatches += 1
                if mismatches <= 5:
                    print(f"MISMATCH trial={trial} q={quantized} d={d}: naive={ka} fast={kb}")
    assert mismatches == 0, f"{mismatches} mismatches out of 800 comparisons"


def test_fast_equals_naive_policy_wrappers():
    rng = np.random.default_rng(7)
    workload = make_workload(rng)
    cluster = make_cluster(rng, n_nodes=20)
    task = TaskClass("t", cpu_milli=1000, memory_mib=100, gpu_demand=0.5)
    for quantized in (False, True):
        a = FGDPolicy(workload, quantized=quantized).decide(cluster, task)
        b = FastFGDPolicy(workload, quantized=quantized).decide(cluster, task)
        ka = (a[0].id, a[1]) if a[0] is not None else None
        kb = (b[0].id, b[1]) if b[0] is not None else None
        assert ka == kb


def test_fast_equals_naive_on_pure_cpu_cluster():
    """Regression for the gmax == 0 crash: CPU-only clusters must not break the fast path."""
    wl = make_workload(np.random.default_rng(3))
    cluster = Cluster([Node("cpu0", 0, cpu_milli=32000), Node("cpu1", 0, cpu_milli=8000)])
    tasks = [
        TaskClass("cpu_only", cpu_milli=1000, gpu_demand=0.0),
        TaskClass("whole", cpu_milli=1000, gpu_demand=1, whole_gpu=True),
        TaskClass("frac", cpu_milli=1000, gpu_demand=0.5),
    ]
    for cls in tasks:
        for quantized in (False, True):
            a_node, a_g = naive_decide(cluster.nodes, wl, cls, quantized=quantized)
            b_node, b_g = fast_decide(cluster.nodes, wl, cls, quantized=quantized)
            ka = (a_node.id, a_g) if a_node is not None else None
            kb = (b_node.id, b_g) if b_node is not None else None
            assert ka == kb, f"q={quantized} cls={cls.name}: {ka} vs {kb}"


def _run_sequential_equivalence(n_tasks):
    from fgdg2.datasets.v2023 import (
        build_workload,
        load_nodes,
        load_pods,
        task_class_from_row,
    )

    nodes_csv = NODES_CSV
    pods_csv = PODS_CSV
    workload, pods = build_workload(load_pods(pods_csv))
    rng = np.random.default_rng(0)
    rows = [pods.iloc[int(rng.integers(len(pods)))] for _ in range(n_tasks)]

    for quantized in (False, True):
        cluster_a = load_nodes(nodes_csv)
        cluster_b = load_nodes(nodes_csv)
        pa, pb = FGDPolicy(workload, quantized=quantized), FastFGDPolicy(workload, quantized=quantized)
        for row in rows:
            cls = task_class_from_row(row)
            na, ga = pa.decide(cluster_a, cls)
            nb, gb = pb.decide(cluster_b, cls)
            ka = (na.id, ga) if na is not None else None
            kb = (nb.id, gb) if nb is not None else None
            assert ka == kb, f"q={quantized} row={row['name']}: {ka} vs {kb}"
            if na is not None:
                na.place(cls, gpu_idx=ga)
                nb.place(cls, gpu_idx=gb)


@needs_trace
@pytest.mark.slow
def test_fast_sequential_equivalence_on_real_trace():
    _run_sequential_equivalence(60)


@needs_trace
@pytest.mark.slow
def test_fast_sequential_equivalence_deep_real_trace():
    """The CPU-boundary bug (§9.3.7) appeared only at task ~76; run deep enough to reach
    boundary states. Kept as `slow`; `make test-all` runs it."""
    _run_sequential_equivalence(700)
