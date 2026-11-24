import numpy as np
import pytest

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.objective import (
    fragmentation_rate,
    node_fragmentation,
    node_workload_fragmentation,
    raw_node_workload_fragmentation,
    raw_node_workload_fragmentation_batch,
    workload_fragmentation,
)
from fgdg2.core.tasks import TaskClass, Workload

DEMANDS = [1, 2, 4, 8]


def make_workload():
    classes = [TaskClass(f"c{k}", cpu_milli=8000, gpu_demand=k, whole_gpu=True) for k in DEMANDS]
    return Workload(classes, np.array([0.4, 0.3, 0.2, 0.1]))


def make_frac_workload():
    classes = [
        TaskClass("f25", cpu_milli=4000, gpu_demand=0.25),
        TaskClass("f50", cpu_milli=4000, gpu_demand=0.5),
    ]
    return Workload(classes, np.array([0.5, 0.5]))


def make_cluster(rng, n_nodes=10, max_gpus=8):
    return Cluster([
        Node(f"n{i}", 0 if rng.random() < 0.1 else int(rng.integers(1, max_gpus + 1)),
             model="T", cpu_milli=32000)
        for i in range(n_nodes)
    ])


def test_fragmentation_cases():
    load = make_workload()
    c8 = load.classes[3]
    c1 = load.classes[0]
    node = Node("a", 4, cpu_milli=32000)
    assert node_fragmentation(node, c8) == pytest.approx(4.0)
    node.place(c1)
    assert node_fragmentation(node, c8) == pytest.approx(3.0)
    node.place(c1)
    node.place(c1)
    node_fragmentation(node, c8)
    assert node_fragmentation(node, c8) == pytest.approx(1.0)
    node.place(c1)
    assert node_fragmentation(node, c8) == pytest.approx(0.0)


def test_cpu_only_case1():
    node = Node("a", 4)
    cpu_cls = TaskClass("cpu", cpu_milli=1000, gpu_demand=0.0)
    assert node_fragmentation(node, cpu_cls) == pytest.approx(4.0)


def test_workload_fragmentation_sums():
    load = make_workload()
    node = Node("a", 4, cpu_milli=32000)
    # k=8 doesn't fit a 4-GPU node (Case 1) -> contributes p8 * total = 0.1 * 4
    assert workload_fragmentation([node], load) == pytest.approx(0.1 * 4.0)
    full_size = Node("b", 8, cpu_milli=32000)
    assert workload_fragmentation([full_size], load) == pytest.approx(0.0)


def test_place_and_release_conservation():
    rng = np.random.default_rng(0)
    cluster = make_cluster(rng)
    load = make_workload()
    placed = []
    n_places = 0
    used_before = 0.0
    for _ in range(2000):
        action = rng.random()
        if placed and action < 0.35:
            i = int(rng.integers(len(placed)))
            node, placement, cls = placed.pop(i)
            used_before -= cls.gpu_demand
            node.release(placement, cls)
            assert cluster.total_used_gpus == pytest.approx(used_before)
            continue
        cls = load.sample(rng)
        feas = cluster.feasible_nodes(cls)
        if not feas:
            continue
        node = feas[rng.integers(len(feas))]
        used_before += cls.gpu_demand
        placement = node.place(cls)
        placed.append((node, placement, cls))
        n_places += 1
        assert cluster.total_used_gpus == pytest.approx(used_before)
    assert n_places >= 100, f"test is vacuous: only {n_places} placements"
    for node, placement, cls in placed:
        node.release(placement, cls)
    assert cluster.total_used_gpus == pytest.approx(0.0)
    for n in cluster.nodes:
        assert (n.gpu_free <= 1.0 + 1e-9).all()
        assert (n.gpu_free >= -1e-9).all()
        assert n.free_slots >= 0


def test_place_release_roundtrip_fractional():
    node = Node.with_free("a", [0.8, 0.95], cpu_milli=32000)
    cls = TaskClass("f025", gpu_demand=0.25)
    placement = node.place(cls, gpu_idx=1)
    assert node.gpu_free.sum() == pytest.approx(1.50)
    node.release(placement, cls)
    assert node.gpu_free.sum() == pytest.approx(1.75)
    assert node.gpu_free[1] == pytest.approx(0.95)
    occupied = Node.with_free("b", [0.9], cpu_milli=32000)
    p1 = occupied.place(TaskClass("f05", gpu_demand=0.5), gpu_idx=0)
    p2 = occupied.place(TaskClass("f04", gpu_demand=0.4), gpu_idx=0)
    assert occupied.gpu_free[0] == pytest.approx(0.0)
    occupied.release(p2, TaskClass("f04", gpu_demand=0.4))
    assert occupied.gpu_free[0] == pytest.approx(0.4)
    occupied.release(p1, TaskClass("f05", gpu_demand=0.5))
    assert occupied.gpu_free[0] == pytest.approx(0.9)


def test_gpu_model_constraint():
    node_t4 = Node("a", 4, model="T4", cpu_milli=32000)
    node_v100 = Node("b", 4, model="V100M32", cpu_milli=32000)
    cls = TaskClass("j", cpu_milli=1000, gpu_demand=1, whole_gpu=True, gpu_spec=("V100M32",))
    assert not node_t4.fits(cls)
    assert node_v100.fits(cls)


def test_workload_validation():
    with pytest.raises(ValueError):
        Workload([], )
    classes = [TaskClass("a", gpu_demand=1, whole_gpu=True), TaskClass("b", gpu_demand=2, whole_gpu=True)]
    with pytest.raises(ValueError):
        Workload(classes, np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        TaskClass("bad", gpu_demand=1.5)


def test_fragmentation_rate():
    load = make_workload()
    nodes = [Node("a", 4, cpu_milli=32000), Node("b", 8, cpu_milli=32000)]
    # F_a = p8 * 4 = 0.4; F_b = 0; total free = 12
    assert workload_fragmentation(nodes, load) == pytest.approx(0.4)
    assert fragmentation_rate(nodes, load) == pytest.approx(0.4 / 12.0)
    for i in range(len(nodes)):
        nodes[i] = Node.with_free(f"z{i}", [0.0] * nodes[i].num_gpus)
    assert fragmentation_rate(nodes, load) == 0.0


def test_frac_workload_fragmentation():
    load = make_frac_workload()
    node = Node.with_free("a", [1.0, 0.6], cpu_milli=32000)
    f = node_workload_fragmentation(node, load)
    assert f == pytest.approx(0.5 * 0.0 + 0.5 * 0.0)
    node2 = Node.with_free("b", [0.3, 0.2], cpu_milli=32000)
    f2 = node_workload_fragmentation(node2, load)
    assert f2 == pytest.approx(0.5 * 0.2 + 0.5 * 0.5)


def test_cluster_fragmentation_decreases_when_filled():
    load = make_workload()
    node = Node("a", 4, cpu_milli=32000)
    f_before = node_fragmentation(node, load.classes[0])
    node.place(load.classes[1])
    f_after = node_fragmentation(node, load.classes[0])
    assert f_before >= f_after


def test_fractional_fit_on_partial_gpu():
    node = Node.with_free("a", [0.5, 0.0], cpu_milli=32000)
    frac = TaskClass("f025", gpu_demand=0.25)
    whole = TaskClass("b1", gpu_demand=1, whole_gpu=True)
    assert node.free_slots == 0
    assert node.fits(frac)
    assert not node.fits(whole)
    node.place(frac, gpu_idx=0)
    assert node.gpu_free[0] == pytest.approx(0.25)
    assert node_fragmentation(node, whole) == pytest.approx(0.25)


def test_workload_sampling_is_seeded():
    load = make_workload()
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    assert [load.sample(r1).name for _ in range(20)] == [load.sample(r2).name for _ in range(20)]


def test_place_whole_honors_explicit_gpu_idx():
    node = Node("a", 4, cpu_milli=32000)
    cls = TaskClass("k2", cpu_milli=1000, gpu_demand=2, whole_gpu=True)
    placement = node.place(cls, gpu_idx=2)
    assert placement.gpu_indices[0] == 2
    assert set(placement.gpu_indices) == {2, 0}
    assert node.gpu_free[2] == 0.0 and node.gpu_free[0] == 0.0 and node.gpu_free[1] == 1.0
    with pytest.raises(ValueError):
        node.place(cls, gpu_idx=99)
    partial = Node.with_free("b", [1.0, 0.5, 1.0], cpu_milli=32000)
    with pytest.raises(ValueError):
        partial.place(TaskClass("k1", gpu_demand=1, whole_gpu=True), gpu_idx=1)


def test_cluster_copy_is_independent():
    cluster = Cluster([Node("a", 4, cpu_milli=32000), Node("b", 2, cpu_milli=8000)])
    clone = cluster.copy()
    cls = TaskClass("k2", cpu_milli=1000, gpu_demand=2, whole_gpu=True)
    clone.node_by_id("a").place(cls)
    assert clone.total_used_gpus == pytest.approx(2.0)
    assert cluster.total_used_gpus == pytest.approx(0.0)
    assert cluster.node_by_id("a").free_slots == 4


def test_with_free_validates_shares():
    with pytest.raises(ValueError):
        Node.with_free("a", [1.5, 0.0])
    with pytest.raises(ValueError):
        Node.with_free("a", [-0.1])
    with pytest.raises(ValueError):
        Node.with_free("a", [float("nan")])


def test_cpu_boundary_scalar_batch_agree_and_fit():
    """Regression for the §9.3.7 bug: cpu_free == class cpu must fit everywhere."""
    wl = Workload(
        [
            TaskClass("a", cpu_milli=16000, gpu_demand=1.0, whole_gpu=True),
            TaskClass("b", cpu_milli=16001, gpu_demand=1.0, whole_gpu=True),
        ],
        np.array([0.5, 0.5]),
    )
    node = Node("n", 2, cpu_milli=16000)
    assert node.fits(wl.classes[0])
    assert not node.fits(wl.classes[1])
    scalar = raw_node_workload_fragmentation(node.gpu_free, wl, node.cpu_free)
    batch = raw_node_workload_fragmentation_batch(
        node.gpu_free[None, :], np.ones((1, 2), dtype=bool), np.array([node.cpu_free]), wl
    )[0]
    assert scalar == pytest.approx(batch)


def test_workload_with_popularities_and_hashable_class():
    load = make_workload()
    new_p = np.array([0.1, 0.2, 0.3, 0.4])
    copy = load.with_popularities(new_p)
    assert copy is not load
    assert copy.popularities == pytest.approx(new_p)
    assert load.popularities == pytest.approx(np.array([0.4, 0.3, 0.2, 0.1]))
    with pytest.raises(ValueError):
        load.with_popularities(np.array([1.0]))
    assert isinstance(hash(load.classes[0]), int)
    assert len({load.classes[0], TaskClass(load.classes[0].name, cpu_milli=load.classes[0].cpu_milli, gpu_demand=load.classes[0].gpu_demand, whole_gpu=True)}) == 1
