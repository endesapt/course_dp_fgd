"""Tracking-M estimator tests — slice semantics, rebuild triggers, fallbacks.

Ported from tests/test_sliding_m.py to the split architecture: estimation
lives in workload/tracking.py providers; DynamicMPolicy only rebuilds the
inner policy when the provider emits a new workload.
"""

import pytest

from fgdg2.core.tasks import workload_from_counts
from fgdg2.items import StreamItem
from fgdg2.policies.dynamic import DynamicMPolicy
from fgdg2.workload.spec import parse_mspec
from fgdg2.workload.tracking import EwmaM, LeadWindowM, QueueAwareM, TrailWindowM


def _item(cpu, gpu, submit):
    return StreamItem(
        gang_id="g", gang_size=1, worker_cpu_milli=cpu, worker_gpu_demand=gpu,
        arrival_index=0, is_gang=False, submit_time_s=submit,
    )


class _Recorder:
    """build_inner stub: records each workload, returns an echo policy."""

    def __init__(self):
        self.workloads = []

    def __call__(self, wl):
        self.workloads.append(wl)
        return type("Inner", (), {
            "wl": wl,
            "decide_gang": lambda self, c, cl, k: ("ok", id(self)),
        })()


def _mix(wl):
    return {c.name: round(float(p), 6) for c, p in
            zip(wl.classes, wl.popularities)}


def test_lead_slice_is_future_window():
    items = [_item(8000, 1, t) for t in (0, 10, 20)] + [_item(4000, 2, 30)]
    rec = _Recorder()
    prov = LeadWindowM(items, window_s=15)
    pol = DynamicMPolicy(rec, prov)
    pol.set_time(0.0)   # slice [0,15): the two gpu=1 items at t=0,10
    assert _mix(rec.workloads[-1]) == {"w8000_1": 1.0}
    pol.set_time(25.0)  # slice [25,40): the gpu=2 item at t=30
    assert _mix(rec.workloads[-1]) == {"w4000_2": 1.0}
    assert prov.rebuilds == 2


def test_lead_evicts_and_holds_last_nonempty():
    items = [_item(8000, 1, 0), _item(8000, 1, 5)]
    rec = _Recorder()
    prov = LeadWindowM(items, window_s=10)
    pol = DynamicMPolicy(rec, prov)
    pol.set_time(0.0)
    pol.set_time(100.0)  # empty slice past the stream end -> keep last M
    assert prov.rebuilds == 1
    assert prov.workload is rec.workloads[-1]


def test_lead_empty_stream_is_safe():
    rec = _Recorder()
    pol = DynamicMPolicy(rec, LeadWindowM([], window_s=10))
    pol.set_time(0.0)          # nothing to emit
    assert rec.workloads == []
    with pytest.raises(RuntimeError):
        pol.decide_gang(None, None, 1)


def test_trail_slice_is_recent_arrivals():
    rec = _Recorder()
    prov = TrailWindowM(window_s=10)
    pol = DynamicMPolicy(rec, prov)
    pol.observe_arrival(_item(8000, 1, 0), now=0.0)
    pol.observe_arrival(_item(4000, 2, 5), now=5.0)
    pol.set_time(5.0)  # (t-10, t] = both items
    assert _mix(rec.workloads[-1]) == {"w8000_1": 0.5, "w4000_2": 0.5}
    pol.observe_arrival(_item(8000, 1, 20), now=20.0)
    pol.set_time(20.0)  # items at 0 and 5 expired; only the t=20 one remains
    assert _mix(rec.workloads[-1]) == {"w8000_1": 1.0}


def test_trail_no_rebuild_on_identical_histogram():
    rec = _Recorder()
    prov = TrailWindowM(window_s=100)
    pol = DynamicMPolicy(rec, prov)
    pol.observe_arrival(_item(8000, 1, 0), now=0.0)
    pol.set_time(1.0)
    pol.set_time(2.0)  # same slice contents -> no rebuild
    assert prov.rebuilds == 1


def test_observe_is_noop_for_lead():
    items = [_item(8000, 1, 0)]
    rec = _Recorder()
    pol = DynamicMPolicy(rec, LeadWindowM(items, window_s=10))
    pol.observe_arrival(_item(4000, 2, 3), now=3.0)  # must NOT enter the lead slice
    pol.set_time(0.0)
    assert _mix(rec.workloads[-1]) == {"w8000_1": 1.0}


def test_decide_forwards_and_stats():
    items = [_item(8000, 1, 0), _item(4000, 2, 5)]
    rec = _Recorder()
    prov = LeadWindowM(items, window_s=100)
    pol = DynamicMPolicy(rec, prov)
    pol.set_time(0.0)
    assert pol.decide_gang(None, None, 1)[0] == "ok"
    assert prov.rebuilds == 1
    stats = pol.tracking_stats()
    assert stats["slide_mode"] == "lead" and stats["slide_rebuilds"] == 1


def test_ewma_converges_to_observed_mix():
    prior = workload_from_counts({(8000, 1.0): 1})
    rec = _Recorder()
    prov = EwmaM(eta=0.5, prior=prior)
    pol = DynamicMPolicy(rec, prov)
    pol.set_time(0.0)  # no arrivals yet -> M = prior
    assert _mix(rec.workloads[-1]) == {"w8000_1": 1.0}
    pol.observe_arrival(_item(4000, 2, 0), now=0.0)
    pol.set_time(0.0)  # one observation at eta=0.5 -> 50/50 mix
    assert _mix(rec.workloads[-1]) == {"w8000_1": 0.5, "w4000_2": 0.5}
    for i in range(1, 10):  # flood with the new class
        pol.observe_arrival(_item(4000, 2, i), now=float(i))
    pol.set_time(9.0)
    mix = _mix(rec.workloads[-1])
    assert mix["w4000_2"] > 0.99  # flooded class dominates
    assert mix["w8000_1"] > 0.0   # prior class decays but keeps support


def test_ewma_never_drops_prior_support():
    prior = workload_from_counts({(8000, 1.0): 1, (4000, 2.0): 1})
    rec = _Recorder()
    prov = EwmaM(eta=0.9, prior=prior)
    pol = DynamicMPolicy(rec, prov)
    pol.observe_arrival(_item(8000, 1, 0), now=0.0)
    pol.set_time(0.0)
    mix = rec.workloads[-1]
    assert len(mix.classes) == 2  # unseen class retains weight 0.1*0.5


def _prior():
    return workload_from_counts({(8000, 1.0): 1.0})


def test_queue_mode_blends_prior():
    """alpha=0.5, queue={gpu1,gpu2} -> 0.5*queue + 0.5*prior(gpu1)."""
    rec = _Recorder()
    prov = QueueAwareM(alpha=0.5, prior=_prior())
    pol = DynamicMPolicy(rec, prov)
    pol.set_queue([_item(8000, 1, 0), _item(4000, 2, 0)])
    mix = _mix(rec.workloads[-1])
    assert mix == {"w8000_1": 0.75, "w4000_2": 0.25}
    assert prov.rebuilds == 1


def test_queue_empty_is_prior_and_dedup():
    rec = _Recorder()
    prov = QueueAwareM(alpha=1.0, prior=_prior())
    pol = DynamicMPolicy(rec, prov)
    pol.set_queue([])                    # empty queue -> pure prior
    assert _mix(rec.workloads[-1]) == {"w8000_1": 1.0}
    pol.set_queue([])                    # unchanged -> no rebuild
    assert prov.rebuilds == 1
    pol.set_queue([_item(4000, 2, 0)])   # queue appears -> pure queue (a=1)
    assert _mix(rec.workloads[-1]) == {"w4000_2": 1.0}
    assert prov.rebuilds == 2


def test_bad_args():
    with pytest.raises(ValueError):
        parse_mspec("slide_sideways:10")
    with pytest.raises(ValueError):
        LeadWindowM([], window_s=0)
    with pytest.raises(ValueError):
        EwmaM(eta=0, prior=_prior())   # eta out of range
    with pytest.raises(ValueError):
        EwmaM(eta=0.5, prior=None)     # missing prior
    with pytest.raises(ValueError):
        QueueAwareM(alpha=0.0, prior=_prior())
    with pytest.raises(ValueError):
        QueueAwareM(alpha=0.5, prior=None)


def test_runner_invokes_hooks():
    """ChurnRunner must call observe_arrival() once per arrival and set_time()
    before every decide — verified via a counting stub policy."""
    from fgdg2.core.cluster import Cluster, Node
    from fgdg2.sim.churn import ChurnRunner

    calls = {"observe": 0, "set_time": 0}

    class _Stub:
        def observe_arrival(self, item, now):
            calls["observe"] += 1

        def set_time(self, now):
            calls["set_time"] += 1

        def decide_gang(self, cluster, cls, k):
            return None  # force queueing -> retries also tick set_time

    node = Node("n", 8, cpu_milli=64000)
    cluster = Cluster(nodes=[node])
    items = [
        StreamItem(gang_id=f"g{i}", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=99.0, arrival_index=i, is_gang=False,
                   submit_time_s=float(i), duration_s=10.0)
        for i in range(3)  # unplaceable demand -> everything queues
    ]
    r = ChurnRunner(cluster, _Stub(), capacity=8.0).run(items)
    assert calls["observe"] == 3
    assert calls["set_time"] >= 3
    assert r.items_total == 3


def test_runner_invokes_set_queue():
    """The queue hook fires before every decide with the items still waiting."""
    from fgdg2.core.cluster import Cluster, Node
    from fgdg2.sim.churn import ChurnRunner

    snapshots = []

    class _Stub:
        def set_queue(self, items, now=0.0):
            snapshots.append(list(items))

        def decide_gang(self, cluster, cls, k):
            return None  # nothing fits -> queue grows

    node = Node("n", 8, cpu_milli=64000)
    cluster = Cluster(nodes=[node])
    items = [
        StreamItem(gang_id=f"g{i}", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=99.0, arrival_index=i, is_gang=False,
                   submit_time_s=float(i), duration_s=10.0)
        for i in range(3)
    ]
    ChurnRunner(cluster, _Stub(), capacity=8.0).run(items)
    # 3 arrivals + scan-mode retries; the queue grows so later snapshots are longer
    assert len(snapshots) >= 3
    assert any(len(s) >= 2 for s in snapshots)
