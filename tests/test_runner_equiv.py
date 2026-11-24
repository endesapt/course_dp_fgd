"""Old-vs-new equivalence: fgdg (legacy, untouched) and fgdg2 (rewrite) are both
importable in this env, so identical inputs must produce identical results —
the safety net that the rewrite preserves the pinned semantics before any
swap. Runtime/wall fields are excluded (wall-clock dependent)."""

import numpy as np
import pytest

fgdg = pytest.importorskip("fgdg", reason="legacy package not installed")

from fgdg.core.cluster import Cluster as Cluster1
from fgdg.core.node import Node as Node1
from fgdg.core.tasks import workload_from_counts as wfc1
from fgdg.policies.gang import DPFGDGang as DP1
from fgdg.sim.churn import ChurnRunner as Churn1
from fgdg.sim.gang import GangRunner as Gang1

from fgdg2.core.cluster import Cluster as Cluster2
from fgdg2.core.node import Node as Node2
from fgdg2.core.tasks import workload_from_counts as wfc2
from fgdg2.policies.gang import DPFGDGang as DP2
from fgdg2.sim.churn import ChurnRunner as Churn2
from fgdg2.sim.gang import GangRunner as Gang2


def _clusters(n_nodes=8):
    rng = np.random.default_rng(7)
    specs = [
        (f"n{i}", int(rng.integers(2, 5)), int(rng.choice([8000, 16000])))
        for i in range(n_nodes)
    ]
    return (
        Cluster1([Node1(i, g, cpu_milli=c) for i, g, c in specs]),
        Cluster2([Node2(i, g, cpu_milli=c) for i, g, c in specs]),
    )


def _items(n=60, seed=5, with_time=False):
    """Identical dict stream — old runner takes dicts, new coerces them."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = int(rng.choice([1, 1, 2, 4]))
        d = {
            "gang_id": f"g{i:06d}", "gang_size": k,
            "worker_cpu_milli": int(rng.choice([1000, 2000, 4000])),
            "worker_gpu_demand": float(rng.choice([1.0, 2.0])),
            "arrival_index": i, "is_gang": k > 1,
        }
        if with_time:
            d["submit_time_s"] = float(i) * 60.0
            d["duration_s"] = float(rng.uniform(30.0, 180.0))
        out.append(d)
    return out


def _diff(a, b, fields):
    return {f: (getattr(a, f), getattr(b, f)) for f in fields
            if getattr(a, f) != getattr(b, f)}


def test_gang_runner_equivalence_dp():
    items = _items()
    c1, c2 = _clusters()
    wl1 = wfc1({(1000, 1.0): 1, (2000, 2.0): 1, (4000, 1.0): 1,
                (1000, 2.0): 1, (2000, 1.0): 1, (4000, 2.0): 1})
    wl2 = wfc2({(1000, 1.0): 1, (2000, 2.0): 1, (4000, 1.0): 1,
                (1000, 2.0): 1, (2000, 1.0): 1, (4000, 2.0): 1})
    r1 = Gang1(c1, DP1(wl1), capacity=1e9).run(items)
    r2 = Gang2(c2, DP2(wl2), capacity=1e9).run(items)
    diffs = _diff(r1, r2, [
        "gangs_total", "gangs_placed", "workers_placed", "singletons_total",
        "singletons_placed", "items_placed", "unplaced_items",
        "cumulative_demand", "unallocated_at_end", "failure_reasons",
    ])
    assert diffs == {}, f"old-vs-new mismatch: {diffs}"


def test_gang_runner_equivalence_fractional_background():
    """MixedPolicy dispatch: fractional singletons go to the background policy."""
    from fgdg.policies.baselines import FirstFitPolicy as FF1
    from fgdg.policies.gang import FirstFitGang as FFG1
    from fgdg.policies.gang import MixedPolicy as M1

    from fgdg2.policies.baselines import FirstFitPolicy as FF2
    from fgdg2.policies.gang import FirstFitGang as FFG2
    from fgdg2.policies.gang import MixedPolicy as M2

    rng = np.random.default_rng(3)
    items = []
    for i in range(80):
        frac = rng.random() < 0.3
        items.append({
            "gang_id": f"g{i:06d}", "gang_size": 1,
            "worker_cpu_milli": 1000,
            "worker_gpu_demand": float(rng.choice([0.25, 0.5])) if frac else 1.0,
            "arrival_index": i, "is_gang": False,
        })
    c1, c2 = _clusters(4)
    r1 = Gang1(c1, M1(FFG1(), FF1()), capacity=1e9).run(items)
    r2 = Gang2(c2, M2(FFG2(), FF2()), capacity=1e9).run(items)
    diffs = _diff(r1, r2, [
        "items_placed", "unplaced_items", "singletons_placed",
        "unallocated_at_end", "cumulative_demand",
    ])
    assert diffs == {}, f"old-vs-new mismatch: {diffs}"


@pytest.mark.parametrize("queue_mode", ["scan", "strict", "blocking"])
def test_churn_runner_equivalence(queue_mode):
    items = _items(n=40, seed=11, with_time=True)
    wl1 = wfc1({(1000, 1.0): 1, (2000, 2.0): 1, (4000, 1.0): 1,
                (1000, 2.0): 1, (2000, 1.0): 1, (4000, 2.0): 1})
    wl2 = wfc2({(1000, 1.0): 1, (2000, 2.0): 1, (4000, 1.0): 1,
                (1000, 2.0): 1, (2000, 1.0): 1, (4000, 2.0): 1})
    c1, c2 = _clusters()
    r1 = Churn1(c1, DP1(wl1), capacity=32.0, queue_mode=queue_mode).run(items)
    r2 = Churn2(c2, DP2(wl2), capacity=32.0, queue_mode=queue_mode).run(items)
    diffs = _diff(r1, r2, [
        "items_total", "gangs_total", "gangs_placed", "singletons_placed",
        "items_placed", "workers_placed", "queued_total", "queued_never_placed",
        "max_queue_depth", "releases", "active_at_end", "queue_at_end",
        "waits", "queued_waits_end", "decide_attempts", "span",
        "congested_time", "empty_node_hours", "empty_node_hours_total",
        "unalloc_gpu_hours", "unalloc_gpu_hours_total",
        "node_max_empty_streak", "nodes_never_used",
    ])
    assert diffs == {}, f"queue={queue_mode} old-vs-new mismatch: {diffs}"
