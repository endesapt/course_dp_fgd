"""Gang policy behaviour tests (AGENTS.md §12.5).

- DP >= atomic_fgd on gang placement rate at identical items/seeds, with a
  non-vacuity search for a strict gap (joint placement beats greedy stranding).
- Property suite: >= 1e5 worker placements through the two-phase commit with
  conservation/atomicity/packed/spread invariants checked after EVERY commit.
- Rollback safety of `commit_gang_plan` and dry-run isolation.
- SequentialGangPolicy equals manual sequential application for each base.
"""

import numpy as np
import pytest
from conftest import GANG_DIR, NODES_CSV, PODS_CSV

from fgdg2.core.cluster import Cluster
from fgdg2.core.gang import Step
from fgdg2.core.node import Node
from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.policies.baselines import BestFitPolicy, FirstFitPolicy, RandomPolicy
from fgdg2.policies.gang import (
    AtomicSequentialFGDGang,
    AtomicSequentialRawFGDGang,
    BestFitGang,
    DPFGDGang,
    FGDIntegerPolicy,
    FirstFitGang,
    MixedPolicy,
    RandomGang,
    RawFGDTaskPolicy,
    SequentialGangPolicy,
)
from fgdg2.sim.commit import commit_gang_plan
from fgdg2.sim.gang import GangRunner


def make_workload(rng, n=2):
    classes = [
        TaskClass(f"w{i}", cpu_milli=int(rng.choice([2000, 4000, 8000])),
                  gpu_demand=float(d), whole_gpu=True)
        for i, d in enumerate(rng.integers(1, 3, size=n))
    ]
    pops = rng.random(n) + 0.1
    return Workload(classes, pops / pops.sum())


def make_cluster(rng, n_nodes=8):
    return Cluster([
        Node(f"n{i}", int(rng.integers(1, 5)), cpu_milli=int(rng.choice([8000, 16000])))
        for i in range(n_nodes)
    ])


def make_items(rng, workload, n_items=25):
    items = []
    for i in range(n_items):
        cls = workload.sample(rng)
        k = int(rng.choice([2, 4, 6])) if rng.random() < 0.4 else 1
        items.append(
            {
                "gang_id": f"g{i:06d}",
                "gang_size": k,
                "worker_cpu_milli": int(cls.cpu_milli),
                "worker_gpu_demand": float(cls.gpu_demand),
                "arrival_index": i,
                "is_gang": k > 1,
            }
        )
    return items


def test_dp_at_least_atomic_placement_rate():
    """Placement rate: DP >= atomic_fgd (spec gate).

    At a GIVEN state the success condition is identical for all spread policies: a
    worker's per-node capacity floors (GPU/CPU/memory) each decrease by exactly 1 per
    placement, so sequential greedy succeeds iff sum(caps) >= k — the same condition
    as DP feasibility (no within-decision stranding). Stream-level rates still differ
    because the policies visit different states; DP's better one-step placements leave
    less future capacity fragmentation (see the demo stream test and
    test_dp_objective_not_worse_than_atomic)."""
    rng = np.random.default_rng(99)
    for _ in range(30):
        workload = make_workload(rng)
        cluster = make_cluster(rng, n_nodes=int(rng.integers(5, 10)))
        items = make_items(rng, workload, n_items=int(rng.integers(15, 30)))
        res_dp = GangRunner(cluster.copy(), DPFGDGang(workload), capacity=10**9).run(items)
        res_at = GangRunner(
            cluster.copy(), AtomicSequentialFGDGang(workload), capacity=10**9
        ).run(items)
        assert res_dp.gangs_placed >= res_at.gangs_placed, (
            f"dp {res_dp.gangs_placed} < atomic {res_at.gangs_placed}"
        )


def _raw_delta(cluster, workload, cls, plan):
    from fgdg2.core.objective import workload_fragmentation

    f_before = workload_fragmentation(cluster.nodes, workload)
    dry = cluster.copy()
    if not commit_gang_plan(dry, cls, plan):
        return None
    return workload_fragmentation(dry.nodes, workload) - f_before


def test_dp_objective_not_worse_than_atomic():
    """Non-vacuity of the DP gain: its one-step plan is never worse under the raw
    FGD objective, and strictly better somewhere (joint placement vs greedy)."""
    rng = np.random.default_rng(99)
    trials = 0
    strict = 0
    for _ in range(40):
        workload = make_workload(rng)
        cluster = make_cluster(rng, n_nodes=int(rng.integers(5, 10)))
        cls = workload.sample(rng)
        k = int(rng.choice([2, 4, 6]))
        plan_dp = DPFGDGang(workload).decide_gang(cluster, cls, k)
        plan_at = AtomicSequentialFGDGang(workload).decide_gang(cluster, cls, k)
        assert (plan_dp is None) == (plan_at is None)
        if plan_dp is None:
            continue
        trials += 1
        d_dp = _raw_delta(cluster, workload, cls, plan_dp)
        d_at = _raw_delta(cluster, workload, cls, plan_at)
        assert d_dp is not None and d_at is not None
        assert d_dp <= d_at + 1e-12
        if d_dp < d_at - 1e-12:
            strict += 1
    assert trials >= 20
    assert strict >= 1, "non-vacuity: DP never strictly improved on atomic"


def test_dp_at_least_atomic_on_committed_demo_streams():
    """Same comparison on a real schema-v2 demo stream (first 400 arrivals of the
    f50 shape-mix stream with fractional background)."""
    from fgdg2.datasets.synth import policy_workload
    from fgdg2.datasets.v2023 import (
        build_m_fractional,
        build_m_whole,
        load_nodes,
        load_pods,
    )
    from fgdg2.streams.io import load_stream

    pods = load_pods(PODS_CSV)
    workload, _ = build_m_whole(pods)
    frac_workload, _ = build_m_fractional(pods)
    nodes_csv = NODES_CSV
    header, items = load_stream(
        GANG_DIR / "synth_gangs_seed0_f50_mix_load1_frac28.jsonl", workload, frac_workload
    )
    items = items[:400]
    assert sum(1 for it in items if it.is_gang) >= 10  # non-vacuity
    policy_wl = policy_workload(workload, header, frac_workload)
    dp = MixedPolicy(DPFGDGang(policy_wl), FirstFitPolicy())
    atomic = MixedPolicy(AtomicSequentialFGDGang(policy_wl), FirstFitPolicy())
    res_dp = GangRunner(load_nodes(nodes_csv), dp, capacity=10**9).run(items)
    res_at = GangRunner(load_nodes(nodes_csv), atomic, capacity=10**9).run(items)
    assert res_dp.gangs_placed >= res_at.gangs_placed


def test_seq_fgd_equals_dp_at_k1():
    """seq_fgd (sequential RAW-FGD) must pick the SAME node as DP for singleton
    placements — same objective, same tie-break — even with fractional classes
    in M, partial shares on nodes, and CPU pressure (the regime where
    atomic_fgd's φ-specialization diverges)."""
    rng = np.random.default_rng(20260927)
    for _ in range(300):
        g = int(rng.choice([2, 4, 8]))
        cpu_cap = int(rng.choice([16000, 64000, 96000]))
        classes = [
            TaskClass(f"w{d}", cpu_milli=int(rng.integers(0, cpu_cap + 1)),
                      gpu_demand=float(d), whole_gpu=True)
            for d in range(1, g + 1)
        ] + [
            TaskClass("f3", cpu_milli=int(rng.integers(0, cpu_cap + 1)), gpu_demand=0.3),
            TaskClass("f6", cpu_milli=int(rng.integers(0, cpu_cap + 1)), gpu_demand=0.6),
        ]
        pops = rng.random(len(classes)) + 0.05
        workload = Workload(tuple(classes), pops / pops.sum())
        nodes = []
        for i in range(int(rng.integers(3, 9))):
            free = np.array([rng.choice([0.0, 0.3, 0.5, 0.7, 1.0]) for _ in range(g)])
            node = Node.with_free(f"n{i}", free, cpu_milli=cpu_cap)
            node._used_cpu = int(rng.integers(0, cpu_cap))
            nodes.append(node)
        cluster = Cluster(nodes)
        cls = classes[int(rng.integers(g))]  # a whole class
        plan_seq = AtomicSequentialRawFGDGang(workload).decide_gang(
            cluster.copy(), cls, 1)
        plan_dp = DPFGDGang(workload).decide_gang(cluster.copy(), cls, 1)
        assert (plan_seq is None) == (plan_dp is None)
        if plan_seq is not None:
            assert plan_seq[0].node_id == plan_dp[0].node_id


def test_seq_fgd_diverges_from_dp_only_on_gangs():
    """For k>=2 the sequential plan may differ from DP's joint plan (that is the
    ablation); DP's realized raw-objective delta is never worse."""
    rng = np.random.default_rng(7)
    strict = 0
    for _ in range(60):
        workload = make_workload(rng)
        cluster = make_cluster(rng, n_nodes=int(rng.integers(4, 8)))
        cls = workload.sample(rng)
        k = int(rng.choice([2, 4]))
        plan_dp = DPFGDGang(workload).decide_gang(cluster, cls, k)
        plan_sq = AtomicSequentialRawFGDGang(workload).decide_gang(cluster, cls, k)
        assert (plan_dp is None) == (plan_sq is None)
        if plan_dp is None:
            continue
        d_dp = _raw_delta(cluster, workload, cls, plan_dp)
        d_sq = _raw_delta(cluster, workload, cls, plan_sq)
        assert d_dp <= d_sq + 1e-12
        if d_dp < d_sq - 1e-12:
            strict += 1
    assert strict >= 1, "non-vacuity: DP never strictly improved on seq_fgd"


def test_seq_fgd_fast_matches_generic():
    """The incremental AtomicSequentialRawFGDGang.decide_gang override must be
    plan-identical to the generic SequentialGangPolicy place/rollback path over
    RawFGDTaskPolicy — same marginal scores at every step (F_n is node-separable,
    so untouched rows never change), same tie-break, same beta semantics."""
    rng = np.random.default_rng(20260927)
    for _ in range(250):
        g = int(rng.choice([2, 4, 8]))
        cpu_cap = int(rng.choice([16000, 64000, 96000]))
        mem_cap = int(rng.choice([0, 4096, 65536]))
        spec = ("a", "b") if rng.random() < 0.3 else None
        classes = [
            TaskClass(
                f"w{d}",
                cpu_milli=int(rng.integers(0, cpu_cap + 1)),
                memory_mib=int(rng.integers(0, mem_cap + 1)) if mem_cap else 0,
                gpu_demand=float(d),
                whole_gpu=True,
                gpu_spec=(("a",) if spec and rng.random() < 0.5 else None),
            )
            for d in range(1, g + 1)
        ] + [
            TaskClass("f3", cpu_milli=int(rng.integers(0, cpu_cap + 1)), gpu_demand=0.3),
            TaskClass("f6", cpu_milli=int(rng.integers(0, cpu_cap + 1)), gpu_demand=0.6),
        ]
        pops = rng.random(len(classes)) + 0.05
        workload = Workload(tuple(classes), pops / pops.sum())
        nodes = []
        for i in range(int(rng.integers(3, 9))):
            free = np.array([rng.choice([0.0, 0.3, 0.5, 0.7, 1.0]) for _ in range(g)])
            node = Node.with_free(
                f"n{i}", free, cpu_milli=cpu_cap, memory_mib=mem_cap,
                model=("a" if spec and rng.random() < 0.7 else "b") if spec else None,
            )
            node._used_cpu = int(rng.integers(0, cpu_cap))
            node._used_memory = int(rng.integers(0, mem_cap)) if mem_cap else 0
            nodes.append(node)
        cluster = Cluster(nodes)
        cls = classes[int(rng.integers(g))]  # a whole class
        k = int(rng.integers(1, 7))
        beta = float(rng.choice([0.0, 0.0, 0.5, 2.0]))
        expected = SequentialGangPolicy(RawFGDTaskPolicy(workload, beta)).decide_gang(
            cluster.copy(), cls, k)
        got = AtomicSequentialRawFGDGang(workload, beta).decide_gang(cluster.copy(), cls, k)
        assert got == expected, f"k={k} beta={beta}"


def _run_property_suite(target_workers):
    rng = np.random.default_rng(4242)
    workload = make_workload(rng)
    weights = np.array([4.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    weights /= weights.sum()
    names = ["firstfit", "bestfit", "random", "atomic_bf", "atomic_ff",
             "atomic_fgd", "seq_fgd", "dp"]
    counts = dict.fromkeys(names, 0)
    spread_seen = False
    workers = 0
    while workers < target_workers:
        cluster = make_cluster(rng, n_nodes=int(rng.integers(4, 9)))
        policies = {
            "firstfit": FirstFitGang(),
            "bestfit": BestFitGang(16000, 4),
            "random": RandomGang(np.random.default_rng(int(rng.integers(10**9)))),
            "atomic_bf": SequentialGangPolicy(BestFitPolicy(16000, 4)),
            "atomic_ff": SequentialGangPolicy(FirstFitPolicy()),
            "atomic_fgd": AtomicSequentialFGDGang(workload),
            "seq_fgd": AtomicSequentialRawFGDGang(workload),
            "dp": DPFGDGang(workload),
        }
        tracked = 0.0
        for _ in range(60):
            cls = workload.sample(rng)
            k = int(rng.choice([1, 1, 2, 4]))
            pidx = int(rng.choice(len(names), p=weights))
            if names[pidx] == "dp" and counts["dp"] > 1500:
                pidx = 0
            name = names[pidx]
            policy = policies[name]
            plan = policy.decide_gang(cluster, cls, k)
            used_before = cluster.total_used_gpus
            if plan is not None:
                assert len(plan) == k
                node_ids = {s.node_id for s in plan}
                if getattr(policy, "packed", False):
                    assert len(node_ids) == 1, f"packed plan spans nodes: {plan}"
                elif len(node_ids) > 1:
                    spread_seen = True
            placed = plan is not None and commit_gang_plan(cluster, cls, plan)
            used_after = cluster.total_used_gpus
            if placed:
                assert abs(used_after - used_before - k * cls.gpu_demand) < 1e-9
                tracked += k * cls.gpu_demand
                counts[name] += k
                workers += k
            else:
                assert abs(used_after - used_before) < 1e-12
            free_shares = 0.0
            for node in cluster.nodes:
                assert (node.gpu_free <= 1.0 + 1e-9).all()
                assert (node.gpu_free >= -1e-9).all()
                assert 0 <= node._used_cpu <= node.cpu_milli
                assert 0 <= node._used_memory <= node.memory_mib
                free_shares += float(node.gpu_free.sum())
            assert abs(cluster.total_gpus - free_shares - tracked) < 1e-9
            if workers >= target_workers:
                break
    assert workers >= target_workers, f"property suite is vacuous: {workers} workers"
    assert spread_seen, "no spread plan spanning multiple nodes observed"
    return counts


def test_property_suite_invariants_fast():
    """>= 1e4 worker placements through the two-phase commit; invariants checked
    after EVERY commit: conservation, no negative shares, atomicity ('all k or
    none'), packed plans single-node, spread plans may span nodes."""
    counts = _run_property_suite(10_000)
    for name, c in counts.items():
        assert c >= 100, f"non-vacuity: policy {name} placed only {c} workers"


@pytest.mark.slow
def test_property_suite_invariants_deep():
    """The >= 1e5 version required by AGENTS §12.7; kept slow for suite runtime."""
    counts = _run_property_suite(100_000)
    for name, c in counts.items():
        assert c >= 1000, f"non-vacuity: policy {name} placed only {c} workers"


def test_commit_gang_plan_rejects_bad_and_rolls_back():
    workload = make_workload(np.random.default_rng(3), n=1)
    cls = workload.classes[0]
    if cls.gpu_demand != 1:
        cls = TaskClass("w", cpu_milli=2000, gpu_demand=1, whole_gpu=True)
    cluster = Cluster([Node("n0", 2, cpu_milli=8000), Node("n1", 1, cpu_milli=8000)])
    used_before = cluster.total_used_gpus
    assert commit_gang_plan(cluster, cls, ()) is False
    assert commit_gang_plan(cluster, cls, (Step("nope", 0),)) is False
    assert commit_gang_plan(cluster, cls, (Step("n0", 99),)) is False
    assert commit_gang_plan(cluster, cls, (Step("n0", 0), Step("n0", 0))) is False
    assert cluster.total_used_gpus == used_before
    # a valid plan succeeds once and only once (dry-run isolation: state changed)
    plan = (Step("n0", 0), Step("n1", 0))
    assert commit_gang_plan(cluster, cls, plan) is True
    assert cluster.total_used_gpus == used_before + 2
    assert commit_gang_plan(cluster, cls, plan) is False
    assert cluster.total_used_gpus == used_before + 2


def _manual_sequential(base, cluster, cls, k):
    dry = cluster.copy()
    steps = []
    for _ in range(k):
        node, gpu = base.decide(dry, cls)
        if node is None:
            return None
        dry.node_by_id(node.id).place(cls, gpu_idx=gpu)
        steps.append(Step(node.id, gpu))
    return tuple(steps)


def test_sequential_gang_matches_manual_application():
    rng = np.random.default_rng(11)
    workload = make_workload(rng)
    cls = workload.classes[0]
    cluster = make_cluster(rng, n_nodes=6)
    pairs = [
        ("fgd_int", FGDIntegerPolicy(workload), FGDIntegerPolicy(workload)),
        ("raw_fgd", RawFGDTaskPolicy(workload), RawFGDTaskPolicy(workload)),
        ("bestfit", BestFitPolicy(16000, 4), BestFitPolicy(16000, 4)),
        ("firstfit", FirstFitPolicy(), FirstFitPolicy()),
        ("random", RandomPolicy(np.random.default_rng(5)), RandomPolicy(np.random.default_rng(5))),
    ]
    for name, base_manual, base_wrapped in pairs:
        expected = _manual_sequential(base_manual, cluster.copy(), cls, 4)
        got = SequentialGangPolicy(base_wrapped).decide_gang(cluster.copy(), cls, 4)
        assert got == expected, name


def test_fgd_integer_rejects_fractional_classes():
    wl = Workload([TaskClass("f", cpu_milli=1000, gpu_demand=0.5)], np.array([1.0]))
    with pytest.raises(ValueError, match="whole-integer"):
        FGDIntegerPolicy(wl)


def test_mixed_policy_dispatches_fractional_to_background():
    cluster = Cluster(
        [
            Node.with_free("a", [0.5, 1.0], cpu_milli=16000),
            Node.with_free("b", [1.0, 1.0], cpu_milli=16000),
        ]
    )
    policy = MixedPolicy(FirstFitGang(), FirstFitPolicy())
    frac = TaskClass("f", cpu_milli=1000, gpu_demand=0.25)
    plan = policy.decide_gang(cluster, frac, 1)
    assert plan == (Step("a", 0),)
    whole = TaskClass("w", cpu_milli=1000, gpu_demand=1, whole_gpu=True)
    plan_whole = policy.decide_gang(cluster, whole, 1)
    assert plan_whole is not None and len(plan_whole) == 1
    with pytest.raises(ValueError, match="out of scope"):
        policy.decide_gang(cluster, frac, 2)


def test_fgd_integer_mixed_workload_uses_whole_subset():
    whole = Workload(
        [TaskClass("w", cpu_milli=2000, gpu_demand=1, whole_gpu=True)], np.array([1.0])
    )
    mixed = Workload(
        [
            TaskClass("w", cpu_milli=2000, gpu_demand=1, whole_gpu=True),
            TaskClass("f", cpu_milli=1000, gpu_demand=0.5),
            TaskClass("f2", cpu_milli=4000, gpu_demand=0.25),
        ],
        np.array([0.6, 0.3, 0.1]),
    )
    cluster = Cluster(
        [Node("n0", 3, cpu_milli=16000), Node("n1", 2, cpu_milli=8000)]
    )
    worker = whole.classes[0]
    assert FGDIntegerPolicy(mixed).decide(cluster, worker) == FGDIntegerPolicy(whole).decide(
        cluster, worker
    )


def test_runner_fractional_background_counters():
    cluster = Cluster([Node("n0", 2, cpu_milli=32000)])
    policy = MixedPolicy(FirstFitGang(), FirstFitPolicy())
    items = [
        {"gang_id": f"g{i}", "gang_size": 1, "worker_cpu_milli": 1000,
         "worker_gpu_demand": d, "arrival_index": i, "is_gang": False}
        for i, d in enumerate([0.5, 0.5, 0.5, 1.0])
    ]
    res = GangRunner(cluster, policy, capacity=10**9).run(items)
    assert cluster.nodes[0].gpu_free == pytest.approx([0.0, 0.5])
    assert res.items_placed == 3
    assert res.unallocated_at_end == pytest.approx(0.5)
    assert res.items_placed + res.unplaced_items == 4
