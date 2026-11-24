"""Layer-2 churn end-to-end: deadline-accurate releases, queue-mode semantics,
wait/metric sanity. These are new tests — the old suite exercised pieces but
never a full scan-vs-strict-vs-blocking divergence scenario."""

from types import SimpleNamespace

import pytest

from fgdg2.core.cluster import Cluster
from fgdg2.core.node import Node
from fgdg2.core.tasks import workload_from_counts
from fgdg2.policies.gang import DPFGDGang
from fgdg2.sim.capacity import CapacityTracker
from fgdg2.sim.churn import ChurnRunner
from fgdg2.sim.elastic import ElasticConfig, ElasticPool
from fgdg2.sim.events import EventHeap

WL = workload_from_counts({(1000, 1.0): 1})
CLS = WL.classes[0]


def _item(i, k, submit, dur, d=1.0):
    return {
        "gang_id": f"g{i}", "gang_size": k, "worker_cpu_milli": 1000,
        "worker_gpu_demand": d, "arrival_index": i,
        "is_gang": k > 1, "submit_time_s": float(submit), "duration_s": float(dur),
    }


def _cluster(n=4):
    return Cluster([Node("n", n, cpu_milli=64000)])


def _run(items, queue_mode="scan", n_gpus=4):
    return ChurnRunner(
        _cluster(n_gpus), DPFGDGang(WL), capacity=float(n_gpus),
        queue_mode=queue_mode).run(items)


def test_release_frees_capacity_for_queued_item():
    """A queued gang must be placed at the RELEASE deadline, not lazily at the
    next arrival (the 2026-09-23 event-loop fix)."""
    items = [
        _item(0, 1, 0, 10),   # placed at 0, holds 1 GPU until t=10
        _item(1, 4, 1, 5),    # needs all 4 -> queues
        _item(2, 1, 20, 5),   # placed at 20
    ]
    res = _run(items)
    assert res.releases == 3
    assert res.items_placed == 3
    waits = sorted(res.waits)
    assert waits == [0.0, 0.0, 9.0]   # g1 placed exactly at t=10
    assert res.span == pytest.approx(25.0)
    assert res.queue_at_end == 0
    assert res.queued_never_placed == 0


def test_deadline_releases_before_same_timestamp_arrival():
    """Release-before-place at equal timestamps: a release at t frees capacity
    that the arrival at t can immediately use."""
    items = [
        _item(0, 4, 0, 10),   # full node, releases at t=10
        _item(1, 4, 10, 5),   # arrives exactly at the release deadline
    ]
    res = _run(items)
    assert res.items_placed == 2
    assert sorted(res.waits) == [0.0, 0.0]  # no queuing: release ran first


def test_queue_modes_diverge_on_head_of_line_blocking():
    """2-GPU node: B (k=2) is stuck behind a partial node; C/D (k=1) behind it.

    - scan:     C placed on arrival (1 free), D placed at t=7 on C's release.
    - strict:   C placed on arrival; D waits behind B (head-of-line stops).
    - blocking: C and D both enqueue behind B without an attempt (global FIFO).
    """
    items = [
        _item(0, 1, 0, 50),   # A: 1 of 2 GPUs, releases at t=50
        _item(1, 2, 1, 5),    # B: needs 2 -> queued
        _item(2, 1, 2, 5),    # C: fits in the 1 free GPU
        _item(3, 1, 3, 10),   # D: behind B in the queue
    ]
    res_scan = _run(items, "scan", n_gpus=2)
    res_strict = _run(items, "strict", n_gpus=2)
    res_block = _run(items, "blocking", n_gpus=2)

    assert res_scan.items_placed == 4
    assert res_scan.queued_never_placed == 0
    # scan waits: A=0, C=0, D=4 (placed at t=7 when C released), B=49 (at t=50)
    assert sorted(res_scan.waits) == [0.0, 0.0, 4.0, 49.0]
    # strict: D sits behind B until B's own release at t=55 frees 2 GPUs
    # (D placed at 55, wait 52) — everything places but waits diverge
    assert res_strict.items_placed == 4
    assert sorted(res_strict.waits) == [0.0, 0.0, 49.0, 52.0]
    # blocking: C and D enqueue behind B without an arrival attempt (global
    # FIFO); both land at t=55 -> waits 53 and 52 on top of B's 49
    assert res_block.items_placed == 4
    assert sorted(res_block.waits) == [0.0, 49.0, 52.0, 53.0]


def test_scan_limit_bounds_queue_retries():
    """queue_scan_limit caps retries per drain — binding only when a release
    frees enough capacity to fit MORE items than the limit. 8-GPU node: A
    (k=8) holds everything until t=100; 10 k=1 items queue behind it.

    - limit 3: at t=100 the drain pops at most 3 -> items 1-3 place; their
      releases at t=110 are three SEPARATE drain events (per-event draining,
      legacy semantics) and each frees 1 GPU + retries 3 -> 4-6, 7-9, 10 all
      land at t=110. Max wait = item 4 (submit t=4): 106.
    - unbounded: one drain at t=100 tries all 10 -> 8 fit, 9/10 queue until
      the first release events at t=110 -> max wait 101 (item 9).
    """
    items = [_item(0, 8, 0, 100)]
    items += [_item(i, 1, i, 10) for i in range(1, 11)]
    res = ChurnRunner(_cluster(8), DPFGDGang(WL), capacity=8.0,
                      queue_mode="scan", queue_scan_limit=3).run(items)
    assert res.items_placed == 11
    assert res.queued_total == 10
    assert res.max_queue_depth == 10
    assert res.waits[0] == 0.0
    assert max(res.waits) == pytest.approx(106.0)

    res2 = ChurnRunner(_cluster(8), DPFGDGang(WL), capacity=8.0,
                       queue_mode="scan").run(items)
    assert max(res2.waits) == pytest.approx(101.0)
    assert res2.decide_attempts > res.decide_attempts  # retries are the cost


def test_metrics_integrals_and_streaks():
    """Time-weighted metrics: empty node accrues hours; placed items break
    the empty streak; conservation holds throughout (assert_consistent ran)."""
    items = [
        _item(0, 1, 0, 10),
        _item(1, 1, 20, 10),
    ]
    res = _run(items)
    # node busy 0-10, empty 10-20, busy 20-30; arrival window ends at t=20
    assert res.empty_node_hours == pytest.approx(10.0)
    # the tail adds only busy time (20->30) — total == window here
    assert res.empty_node_hours_total == pytest.approx(10.0)
    assert res.node_max_empty_streak["n"] == pytest.approx(10.0)
    assert res.nodes_never_used == 0


def test_empty_stream_and_missing_durations():
    assert _run([]).items_total == 0
    with pytest.raises(ValueError, match="duration_s"):
        _run([{"gang_id": "g", "gang_size": 1, "worker_cpu_milli": 1000,
               "worker_gpu_demand": 1.0, "arrival_index": 0, "is_gang": False,
               "submit_time_s": 0.0}])
    with pytest.raises(ValueError, match="queue_mode"):
        ChurnRunner(_cluster(), DPFGDGang(WL), capacity=4.0,
                    queue_mode="chaotic")


# -- elasticity: evict-idle>T + reactive scale-up -------------------------------


def _elastic(items, n_blueprints=1, gpus=4, evict_s=100.0, latency_s=50.0,
             max_nodes=0, reserve=0, warm=False, top_n=0,
             pin_hold=21600.0, bind=False):
    bps = [Node(f"b{i}", gpus, cpu_milli=64000) for i in range(n_blueprints)]
    pool = ElasticPool(ElasticConfig(
        evict_after_s=evict_s, launch_latency_s=latency_s,
        max_nodes=max_nodes, cold_start=not warm, reserve_empty=reserve,
        reserve_top_n=top_n, pin_max_hold_s=pin_hold,
        provision_binding=bind), bps)
    cluster = Cluster([]) if not warm else Cluster(
        [Node(n.id, n.num_gpus, model=n.model,
              cpu_milli=n.cpu_milli, memory_mib=n.memory_mib) for n in bps])
    return ChurnRunner(cluster, DPFGDGang(WL), capacity=float(gpus * n_blueprints),
                       elastic=pool).run(items)


def test_elastic_cold_start_launch_latency_is_wait():
    """Cold pool: the first item queues, a node launches, lands at +delta and
    the wait IS the latency — billing starts at ready, not at request."""
    res = _elastic([_item(0, 1, 0, 20), _item(1, 1, 300, 10)])
    assert res.items_placed == 2
    assert sorted(res.waits) == [50.0, 50.0]
    assert res.launches == 2 and res.evictions == 2
    # node1 online 50->170 (evict at empty 70 + T 100); window ends at 300
    assert res.billed_node_hours == pytest.approx(120.0)
    # node2 online 350->460 adds to the drain-tail total only
    assert res.billed_node_hours_total == pytest.approx(230.0)


def test_elastic_evict_fires_at_empty_plus_T():
    """A node empty at 70 with T=100 dies at 170 — an arrival at 140 still
    finds it online (3 free GPUs); after its release the clock re-arms."""
    items = [_item(0, 1, 0, 20), _item(1, 1, 140, 5)]
    res = _elastic(items)
    assert sorted(res.waits) == [0.0, 50.0]
    assert res.launches == 1 and res.evictions == 1
    # online 50->245 (g1 release at 145 re-arms the check to 245)
    assert res.billed_node_hours == pytest.approx(90.0)   # window ends at 140
    assert res.billed_node_hours_total == pytest.approx(195.0)


def test_stale_evict_check_is_a_noop():
    """Node re-emptied between arming and firing: the OLD check's `since`
    token no longer matches empty_since — it must not evict mid-interval."""
    items = [_item(0, 4, 0, 70),   # lands at 50, releases at 120 -> check@220
             _item(1, 1, 130, 10)]  # uses node 130->140 -> new check@240
    res = _elastic(items)
    assert res.evictions == 1 and res.launches == 1
    # evict happened at 240 (not 220): online 50->240
    assert res.billed_node_hours_total == pytest.approx(190.0)


def test_supply_cap_bounds_launches():
    """Launches stop at the cap, and a gang needing more nodes than the pool
    ever has is never launched for and never placed."""
    items = [_item(0, 4, 0, 500),    # whole node1
             _item(1, 4, 0, 500),    # whole node2 — fills the cap
             _item(2, 12, 0, 500)]   # needs 3 nodes > cap 2: impossible
    res = _elastic(items, n_blueprints=2)
    assert res.launches == 2            # capped, never a third
    assert res.items_placed == 2
    assert res.queued_never_placed == 1


def test_reserve_floor_keeps_last_empty():
    """keep-R-hot: the last R empty nodes survive their eviction check —
    structural headroom for whole-node arrivals."""
    items = [_item(0, 4, 0, 10), _item(1, 1, 0, 10)]
    res = _elastic(items, n_blueprints=2, reserve=1)
    assert res.launches == 2 and res.evictions == 1
    # the surviving node bills until the heap drains (last check at ~210)
    assert res.billed_node_hours_total == pytest.approx(220.0)


def test_never_feasible_item_launches_nothing():
    """d=8 worker on a 4-GPU blueprint pool can never place — the trigger
    must not burn launches on it."""
    it = _item(0, 1, 0, 10)
    it["worker_gpu_demand"] = 8.0
    res = _elastic([it])
    assert res.launches == 0 and res.queued_never_placed == 1


def test_eviction_regret_counts_quick_relaunch():
    """Relaunch within ~delta of an eviction = the timeout guessed wrong."""
    items = [_item(0, 4, 0, 20),   # node dies at 80 (empty 70 + T 10)
             _item(1, 4, 100, 10)]  # relaunches at 100, 20s after the evict
    res = _elastic(items, evict_s=10.0)
    assert res.evictions == 2 and res.eviction_regret == 1


def test_warm_start_static_billing_equivalence():
    """Warm pool + T=inf is the static fleet: no evictions, billed ==
    n_nodes x window — the non-elastic regression anchor."""
    items = [_item(0, 1, 0, 10), _item(1, 1, 100, 10)]
    res = _elastic(items, n_blueprints=1, evict_s=1e12, warm=True)
    assert res.launches == 0
    # the (far-future) eviction still fires in the drain tail — window billing
    # is the static-equivalent quantity
    assert res.billed_node_hours == pytest.approx(100.0)
    static = _run(items)
    assert res.billed_node_hours == pytest.approx(
        static.billed_node_hours)


def test_set_running_sees_only_active_payloads():
    """Infra events share the heap with _Active releases — the running-set
    snapshot must filter on isinstance or it crashes on .receipts/.deadline."""
    seen = []

    class Spy:
        def set_running(self, now, running):
            seen.append((now, list(running)))

        def decide_gang(self, cluster, cls, k):
            return None

    pool = ElasticPool(ElasticConfig(evict_after_s=50.0, launch_latency_s=10.0),
                       [Node("b", 4, cpu_milli=64000)])
    items = [_item(0, 1, 0, 10), _item(1, 1, 100, 10)]
    res = ChurnRunner(Cluster([]), Spy(), capacity=4.0, elastic=pool).run(items)
    # decides ran while _NodeReady/_EvictCheck payloads sat in the heap —
    # the snapshot delivered only (t_start, item) tuples, never crashing
    assert seen and all(len(r) == 0 for _, r in seen)
    assert res.queued_never_placed == 2


def test_pin_masking_changes_fate():
    """M1 reservation: gB (k=2, d=4 — needs 2 wholly-empty nodes) pins the
    free node so gC (k=1, d=1) can't nibble it apart. Without pins gC steals
    it and gB starves ~100s; with pins gB lands at the first release."""
    items = [_item(0, 1, 0, 20, d=4.0),    # gA fills node0 until t=20
             _item(1, 2, 1, 100, d=4.0),   # gB needs 2 empty nodes — pins n1
             _item(2, 1, 2, 100, d=1.0)]   # gC would eat 1 GPU of n1
    res_pin = _elastic(items, n_blueprints=2, evict_s=1e12, warm=True, top_n=1)
    res_nop = _elastic(items, n_blueprints=2, evict_s=1e12, warm=True, top_n=0)
    assert res_pin.items_placed == 3
    assert sorted(res_pin.waits) == [0.0, 19.0, 118.0]
    assert res_nop.items_placed == 3
    assert sorted(res_nop.waits) == [0.0, 0.0, 101.0]  # gB starved by gC


def test_pinned_node_survives_eviction():
    """A pinned node is exempt from idle>T eviction — its idle clock is owned
    by the reservation, not the timeout. Without pins the node dies and the
    gang can never assemble."""
    items = [_item(0, 1, 0, 20, d=4.0),    # gA holds node0 until t=20
             _item(1, 2, 1, 100, d=4.0)]   # gB pins node1 at t=1
    res_pin = _elastic(items, n_blueprints=2, evict_s=10.0, warm=True, top_n=1)
    res_nop = _elastic(items, n_blueprints=2, evict_s=10.0, warm=True, top_n=0)
    # n1 survives its t=10 check (pinned) -> gB places at the t=20 release;
    # both nodes evict later at t=130 once the work is done — that's correct
    assert sorted(res_pin.waits) == [0.0, 19.0]
    assert res_pin.evictions == 2
    assert res_nop.items_placed == 1            # gB starves: node1 evicted@10
    assert res_nop.queued_never_placed == 1


def test_release_pins_rearms_with_original_idle_clock():
    """Regression (bug 3.1): a pinned node keeps its ORIGINAL empty_since
    clock while pinned — release_pins must re-arm the evict check with THAT
    token. The old code armed check(now+T, since=now); the token never
    matched empty_since, the check fired stale, and the freed node
    billed-empty until reused."""
    pool = ElasticPool(ElasticConfig(evict_after_s=100.0),
                       [Node("b", 4, cpu_milli=64000)])
    tracker = CapacityTracker(Cluster([Node("n0", 4, cpu_milli=64000)]))
    heap = EventHeap()
    # n0 empty since t=0 (tracker init). Pin it, hold past the deadline,
    # release while still empty -> overdue: fires AT the release time.
    pool.pins["g"] = {"n0"}
    pool._pin_since["g"] = 0.0
    pool.reserved_nids.add("n0")
    pool.release_pins("g", 200.0, tracker, heap)
    dl, payload = heap.pop()
    assert dl == 200.0                       # overdue deadline, not 200+T
    assert payload.since == 0.0              # ORIGINAL clock, not re-stamped
    assert pool.evict_check(payload, dl, tracker)      # matches -> evicts
    assert tracker.empty_since == {}


def test_released_pin_before_deadline_keeps_original_deadline():
    """Twin case: released BEFORE empty_since+T -> evicts at the ORIGINAL
    deadline (no free idle-time extension from the pin)."""
    pool = ElasticPool(ElasticConfig(evict_after_s=100.0),
                       [Node("b", 4, cpu_milli=64000)])
    tracker = CapacityTracker(Cluster([Node("n0", 4, cpu_milli=64000)]))
    heap = EventHeap()
    pool.pins["g"] = {"n0"}
    pool.reserved_nids.add("n0")
    pool.release_pins("g", 60.0, tracker, heap)
    dl, payload = heap.pop()
    assert dl == 100.0                       # original deadline, not 160
    assert payload.since == 0.0
    assert pool.evict_check(payload, dl, tracker)


def test_pin_expiry_embargoes_repin():
    """pin_max_hold is a real expiry: the released gang is embargoed from
    re-pinning for one hold period, so a second queued gang can take the
    freed nodes — the old code re-pinned the same gang in the same call,
    making pin_max_hold a no-op and holding masked capacity forever."""
    pool = ElasticPool(ElasticConfig(
        reserve_top_n=1, pin_max_hold_s=100.0, evict_after_s=1e12,
        max_nodes=2),
        [Node("b", 4, cpu_milli=64000)])
    tracker = CapacityTracker(Cluster(
        [Node(f"n{i}", 4, cpu_milli=64000) for i in range(2)]))
    heap = EventHeap()
    q_b = SimpleNamespace(item=SimpleNamespace(gang_id="gB"), cls=CLS, k=8)
    q_d = SimpleNamespace(item=SimpleNamespace(gang_id="gD"), cls=CLS, k=8)
    pool.assign_pins([q_b], 1.0, tracker, heap)
    assert pool.pins["gB"] == {"n0", "n1"}   # quota 2 on 4-GPU nodes
    # expiry pass at t=200: gB drops + embargoed -> gD takes the top-N slot
    pool.assign_pins([q_b, q_d], 200.0, tracker, heap)
    assert "gB" not in pool.pins
    assert pool.pins["gD"] == {"n0", "n1"}
    # embargo lifts after one hold period: once gD's pins release, gB can
    # pin again — expiry is a breathing window, not a permanent ban
    pool.release_pins("gD", 301.0, tracker, heap)
    pool.assign_pins([q_b, q_d], 301.0, tracker, heap)
    assert pool.pins["gB"] == {"n0", "n1"}


def test_provision_binding_masks_bound_node():
    """Kueue ProvReq emulation: a node launched FOR g1 lands bound to it —
    g2's arrival can't steal it and must wait for real capacity. Unbound,
    the same node is free-pool: g2 walks in at wait=0 between the ready
    event and g1's next drain — the theft path the binding closes."""
    items = [_item(0, 1, 0, 120, d=4.0),   # g0: queues (cold) -> bound node,
             # places at +50, releases at 170 -> frees node0 for the drain
             _item(1, 2, 10, 30, d=4.0),   # g1 needs 2 empty nodes; cap room
             # leaves it ONE bound launch -> still queued when it lands
             _item(2, 1, 65, 10, d=1.0)]   # g2: arrives while g1's node sits
    res_bnd = _elastic(items, n_blueprints=2, evict_s=1e12, bind=True)
    res_unb = _elastic(items, n_blueprints=2, evict_s=1e12, bind=False)
    assert res_bnd.items_placed == 3
    assert sorted(res_bnd.waits) == [50.0, 135.0, 160.0]  # g2 waits to 200
    assert sorted(res_unb.waits) == [0.0, 50.0, 160.0]   # g2 steals at t=65


def test_bound_node_of_gone_item_arrives_free():
    """Bound launch whose item already placed (left the queue before
    node_ready): the node must land UNBOUND — a zombie pin would leak
    masked capacity forever."""
    pool = ElasticPool(ElasticConfig(
        launch_latency_s=50.0, provision_binding=True),
        [Node("b", 4, cpu_milli=64000)])
    tracker = CapacityTracker(Cluster([]))
    heap = EventHeap()
    pool.on_enqueue("gA", CLS, 1, 0.0, tracker, heap)
    assert pool.launches == 1
    dl, payload = heap.pop()
    assert pool.node_ready(payload, dl, tracker, queue=[]) is True
    nid = payload.node.id
    assert nid in tracker.empty_since        # joined the pool normally
    assert nid not in pool.reserved_nids     # ...but bound to nobody


def test_provision_binding_coverage_is_per_item():
    """Unbound: pending launches count toward ANY queued item's need (they
    join the free pool). Bound: capacity provisioned for gA is spoken for,
    so gB requests its own — per-Workload ProvReq accounting."""
    def mkpool(bind):
        return ElasticPool(ElasticConfig(
            launch_latency_s=50.0, provision_binding=bind),
            [Node(f"b{i}", 4, cpu_milli=64000) for i in range(4)])
    tracker = CapacityTracker(Cluster([]))
    heap = EventHeap()
    unb = mkpool(False)
    unb.on_enqueue("gA", CLS, 8, 0.0, tracker, heap)   # needs 2 nodes
    unb.on_enqueue("gB", CLS, 8, 1.0, tracker, heap)   # pending covers it
    assert unb.launches == 2
    bnd = mkpool(True)
    bnd.on_enqueue("gA", CLS, 8, 0.0, tracker, heap)
    bnd.on_enqueue("gB", CLS, 8, 1.0, tracker, heap)   # gets its OWN 2
    assert bnd.launches == 4


def test_assembly_starvation_metric_counts_fragmented_queue_gpu_time():
    """Enough raw free GPUs for a k=3, d=2 gang (6 GPUs), but each node has an
    odd number free: caps are 1+1+0 < 3, so no feasible split exists. The gang sits from t=1 to the t=10 release: 6 GPU ×
    9 s = 54 GPU-s of assembly starvation."""
    wl2 = workload_from_counts({(1000, 2.0): 1})
    cluster = Cluster([
        Node("n0", 1, cpu_milli=64000),
        Node("n1", 3, cpu_milli=64000),
        Node("n2", 3, cpu_milli=64000),
    ])
    items = [
        _item(0, 1, 0, 10, d=1.0),   # fills n0, releases at t=10
        _item(1, 3, 1, 10, d=2.0),   # 6 GPUs free in aggregate, no split fits
    ]
    res = ChurnRunner(cluster, DPFGDGang(wl2), capacity=7.0).run(items)
    assert res.assembly_starved_events == 1
    assert res.assembly_starved_gpu_s == pytest.approx(54.0)
    assert res.queue_gpu_s == pytest.approx(54.0)
