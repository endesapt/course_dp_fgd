"""Layer-2 churn runner: arrivals + releases + FCFS queue.

Layer 1 (GangRunner) is a closed-loop fill-up: items are placed once, never
leave, failures are permanent. Layer 2 adds the lifecycle the trace actually
has — each item carries submit_time_s + duration_s; placed items release their
resources at submit+duration; failed items wait in a queue and retry on
release events. Durations/queueing are SIMULATION-ONLY evaluation conditions
(§4.7) — the theory model stays duration-free.

Queue modes (the FCFS semantics under test):
- "scan"     — arrivals get one immediate attempt; failures enqueue. On each
               release batch the queue is scanned in FCFS order and every item
               that now fits is placed (skipped items keep their position —
               light backfill, the realistic default).
- "strict"   — same arrivals; on release the queue drains from the head and
               STOPS at the first item that still doesn't fit (head-of-line
               blocking inside the queue).
- "blocking" — global FIFO: if the queue is non-empty, new arrivals enqueue
               WITHOUT an attempt; the queue drains head-only. The pathological
               ablation — one big gang can starve the whole stream.

Release ordering is owned by `sim.events.EventHeap`: releases fire at their
true deadlines BETWEEN arrivals (the post-2026-09-23 fix — lazy-flushing at
the next arrival inflated waits and undercounted empty time).

Tracking-M policies are driven through the ArrivalObserver/TimeAware/
QueueAware hooks (policies without them are unaffected — getattr no-op).
"""

import time
from collections import Counter, deque
from dataclasses import dataclass

from ..core.cluster import Cluster
from ..items import as_items
from ..metrics.results import ChurnResult, gpu_demand_bucket
from .capacity import CapacityTracker
from .commit import _commit_gang_plan
from .elastic import _EvictCheck, _NodeReady
from .events import EventHeap

QUEUE_MODES = ("scan", "strict", "blocking")


@dataclass
class _Queued:
    item: object  # StreamItem
    cls: object   # TaskClass
    k: int
    t_submit: float
    duration: float

    @property
    def class_key(self):
        return self.item.class_key


@dataclass
class _Active:
    deadline: float
    receipts: list          # [(node, placement)]
    cls: object             # TaskClass
    t_start: float          # admit time — held-so-far = now - t_start
    item: object            # StreamItem (running-set estimator snapshot)


class ChurnRunner:
    """Event-driven gang scheduler replay with departures."""

    def __init__(self, cluster, policy, capacity: float, queue_mode: str = "scan",
                 queue_scan_limit: int = 256, elastic=None):
        if queue_mode not in QUEUE_MODES:
            raise ValueError(f"queue_mode must be one of {QUEUE_MODES}")
        self.cluster = cluster
        self.policy = policy
        self.capacity = float(capacity)
        self.queue_mode = queue_mode
        # ElasticPool | None — node supply controller; None = static fleet
        self._elastic = elastic
        # bound decide calls per drain in "scan" mode (FCFS order preserved:
        # only the first K queued items are retried per release batch)
        self.queue_scan_limit = int(queue_scan_limit)
        # queued GPU-demand bookkeeping: updated on enqueue/dequeue so both the
        # timeline snapshots and the time integrals see the queue's COMPOSITION
        self._queue_gpu = 0.0
        self._queue_gpu_b: Counter = Counter()
        # assembly starvation: queued GPU demand whose blocker is fragmentation
        # (aggregate capacity exists, but no feasible gang placement)
        self._starved_gpu = 0.0
        self._packed = bool(getattr(policy, "packed", False))

    def _is_assembly_starved(self, q: _Queued) -> bool:
        """Queued gang cannot assemble even though the cluster has enough RAW
        free GPU shares in aggregate. This is broader than failure_reason's
        capacity_fragmented: it also catches per-node floor loss (e.g. two
        nodes with 3 free GPUs cannot host a k=3, d=2 gang)."""
        if q.cls.gpu_demand < 1.0:
            return False
        d = int(q.cls.gpu_demand)
        demand = q.k * d
        total_free = sum(n.free_slots for n in self.cluster.nodes)
        if total_free < demand:
            return False
        caps = [n.max_whole_workers(q.cls) for n in self.cluster.nodes]
        if self._packed:
            return max(caps, default=0) < q.k
        return sum(caps) < q.k

    def _set_starved(self, q: _Queued, starved: bool) -> None:
        old = bool(getattr(q, "starved", False))
        if old == starved:
            return
        q.starved = starved
        d = q.item.gpu_demand_total
        self._starved_gpu += d if starved else -d

    # -- decide ---------------------------------------------------------------
    def _try_place(self, q: _Queued, res: ChurnResult, now: float,
                   tracker: CapacityTracker) -> _Active | None:
        # tracking-M hooks (policies that lack them are unaffected): set_time
        # advances time-sliced M; set_queue hands over the FCFS queue contents
        # (items waiting to be placed, excluding the one being decided)
        set_time = getattr(self.policy, "set_time", None)
        if set_time is not None:
            set_time(now)
        set_queue = getattr(self.policy, "set_queue", None)
        if set_queue is not None:
            set_queue(list(self._queue), now)
        set_running = getattr(self.policy, "set_running", None)
        if set_running is not None:
            # deadline > now filters out releases already due but still sitting
            # in the heap under pop_due's lazy iteration; isinstance keeps
            # elastic-pool infra payloads (_NodeReady/_EvictCheck) out — they
            # carry no .receipts/.item
            set_running(now, [(a.t_start, a.item) for a in self._heap.payloads()
                              if isinstance(a, _Active) and a.deadline > now])
        # M_t trajectory for tracking arms: hourly snapshot of the estimate
        # the policy is scoring against (bounded size; postmortem evidence)
        prov = getattr(self.policy, "provider", None)
        if (prov is not None and prov.workload is not None
                and (not res.m_history
                     or now - res.m_history[-1][0] >= 3600)):
            res.m_history.append(
                [now, [[c.cpu_milli, c.gpu_demand, float(p)]
                       for c, p in zip(prov.workload.classes,
                                       prov.workload.popularities)]])
        t0 = time.perf_counter()
        # M1 mask: nodes pinned for OTHER gangs are invisible to this decide —
        # same Node objects, so committing the plan on the real cluster works
        cluster = self.cluster
        if self._elastic is not None and self._elastic.reserved_nids:
            hidden = (self._elastic.reserved_nids
                      - self._elastic.pins.get(q.item.gang_id, set()))
            if hidden:
                cluster = Cluster(
                    [n for n in self.cluster.nodes if n.id not in hidden])
        plan = self.policy.decide_gang(cluster, q.cls, q.k)
        res.decide_runtime_ms.append((time.perf_counter() - t0) * 1e3)
        res.decide_attempts += 1
        if plan is None:
            return None
        plan_nodes, emptied, full_consumed = tracker.inspect_plan(q.cls, plan)
        receipts = _commit_gang_plan(self.cluster, q.cls, plan)
        if receipts is None:
            return None
        tracker.place(q.cls, receipts, plan_nodes, emptied, full_consumed, now)
        return _Active(deadline=now + q.duration, receipts=receipts, cls=q.cls,
                       t_start=now, item=q.item)

    def _drain_queue(self, res: ChurnResult, now: float,
                     tracker: CapacityTracker, heap: EventHeap) -> None:
        if not self._queue:
            return
        if self.queue_mode == "scan":
            # pop at most scan_limit items from the head, try each in FCFS
            # order, push survivors back in order — items beyond the limit are
            # never touched (O(K) per drain, not O(queue))
            survivors = []
            for _ in range(min(self.queue_scan_limit, len(self._queue))):
                q = self._queue.popleft()
                act = self._try_place(q, res, now, tracker)
                if act is None:
                    self._set_starved(q, self._is_assembly_starved(q))
                    survivors.append(q)
                    self._signal_queue(q, now, tracker, heap)
                else:
                    self._admit(q, act, res, now, heap, tracker)
                    self._dequeue(q)
            self._queue.extendleft(reversed(survivors))
        else:  # strict / blocking: head-of-line
            while self._queue:
                q = self._queue[0]
                act = self._try_place(q, res, now, tracker)
                if act is None:
                    self._set_starved(q, self._is_assembly_starved(q))
                    self._signal_queue(q, now, tracker, heap)
                    break
                self._queue.popleft()
                self._admit(q, act, res, now, heap, tracker)
                self._dequeue(q)
        # pins are assigned once the queue is back intact — a mid-drain
        # survivor isn't in self._queue yet and would miss its own pass
        if self._elastic is not None:
            self._elastic.assign_pins(self._queue, now, tracker, heap)

    def _dequeue(self, q) -> None:
        self._set_starved(q, False)
        d = q.item.gpu_demand_total
        self._queue_gpu -= d
        b = gpu_demand_bucket(d)
        self._queue_gpu_b[b] -= d
        if self._queue_gpu_b[b] <= 0:
            del self._queue_gpu_b[b]

    def _admit(self, q, act, res, now, heap: EventHeap,
               tracker: CapacityTracker) -> None:
        heap.push(act.deadline, act)
        self._active += 1
        if self._elastic is not None:
            # the gang placed (on its pins or elsewhere) — reservation served
            self._elastic.release_pins(q.item.gang_id, now, tracker, heap)
        res.items_placed += 1
        res.workers_placed += q.k
        w = now - q.t_submit
        res.waits.append(w)
        res.wait_records.append((w, q.item.gpu_demand_total, q.k,
                                 bool(q.item.is_gang), q.t_submit))
        res.wait_gpu_s += w * q.item.gpu_demand_total
        res.wait_gpu_demand += q.item.gpu_demand_total
        if q.item.is_gang:
            res.gangs_placed += 1
        else:
            res.singletons_placed += 1

    def _enqueue(self, q, res, now, tracker, heap) -> None:
        self._queue.append(q)
        d = q.item.gpu_demand_total
        self._queue_gpu += d
        self._queue_gpu_b[gpu_demand_bucket(d)] += d
        res.queued_total += 1
        res.max_queue_depth = max(res.max_queue_depth, len(self._queue))
        self._set_starved(q, self._is_assembly_starved(q))
        if getattr(q, "starved", False):
            res.assembly_starved_events += 1
        self._signal_queue(q, now, tracker, heap)
        if self._elastic is not None:
            # a just-failed gang can pin ALREADY-empty nodes right away —
            # don't wait for the next empty event to start assembling
            self._elastic.assign_pins(self._queue, now, tracker, heap)

    def _signal_queue(self, q, now, tracker, heap) -> None:
        """Tell the elastic pool about unplaceable demand — the reactive
        scale-up trigger (no-op under a static fleet)."""
        if self._elastic is not None:
            self._elastic.on_enqueue(q.item.gang_id, q.cls, q.k, now,
                                     tracker, heap)

    def _dispatch(self, payload, dl, res, tracker, heap) -> None:
        """One heap event at its deadline: job release or pool infrastructure."""
        if isinstance(payload, _Active):
            newly_empty = tracker.release(payload.cls, payload.receipts, dl)
            res.releases += 1
            self._active -= 1
            if self._elastic is not None:
                self._elastic.on_emptied(newly_empty, dl, heap)
                # freed nodes go to reservations BEFORE the queue drains
                self._elastic.assign_pins(self._queue, dl, tracker, heap)
            self._drain_queue(res, dl, tracker, heap)
        elif isinstance(payload, _NodeReady):
            if self._elastic.node_ready(payload, dl, tracker,
                                        queue=self._queue):
                # a fresh node is EMPTY — arm its own idle clock, then let
                # reservations, then the queue, have first pick at it
                self._elastic.on_emptied([payload.node.id], dl, heap)
                self._elastic.assign_pins(self._queue, dl, tracker, heap)
                self._drain_queue(res, dl, tracker, heap)
        elif isinstance(payload, _EvictCheck):
            self._elastic.evict_check(payload, dl, tracker)

    # -- main loop --------------------------------------------------------------
    def run(self, items) -> ChurnResult:
        items = as_items(items)  # public boundary tolerates dicts; schema-checked
        if not items:
            return ChurnResult()
        # event order = submission order (stable sort keeps file order on ties);
        # normalize origin so mid-trace windows don't inflate span/waits
        items.sort(key=lambda it: it.t_submit)
        t0 = min(it.t_submit for it in items)

        res = ChurnResult()
        self._queue = deque()
        self._queue_gpu = 0.0
        self._queue_gpu_b = Counter()
        self._starved_gpu = 0.0
        tracker = CapacityTracker(self.cluster)
        heap = EventHeap()
        self._heap = heap  # running-set snapshot source for set_running hooks
        self._active = 0   # running jobs — len(heap) would count infra events
        if self._elastic is not None:
            self._elastic.arm_initial(tracker, heap)  # warm-start idle clocks
        now = prev = 0.0

        def integrate(dt: float) -> None:
            res.congested_time += dt * float(len(self._queue) > 0)
            res.queue_gpu_s += dt * self._queue_gpu
            res.assembly_starved_gpu_s += dt * self._starved_gpu
            for b, v in self._queue_gpu_b.items():
                res.queue_gpu_s_by_bucket[b] += dt * v

        def snap(kind, extra=None):
            res.timeline.append(
                {
                    "t": now, "kind": kind,
                    "unallocated_shares": tracker.unallocated,
                    "fully_free": tracker.fully_free,
                    "queue_depth": len(self._queue),
                    "queue_gpu": self._queue_gpu,
                    "assembly_starved_gpu": self._starved_gpu,
                    "active": self._active,
                    "empty_nodes": tracker.empty_nodes,
                    "online_nodes": tracker.online_nodes,
                    "pending_nodes": (len(self._elastic.pending)
                                      if self._elastic is not None else 0),
                    "pinned_nodes": (len(self._elastic.reserved_nids)
                                     if self._elastic is not None else 0),
                    **(extra or {}),
                }
            )

        for item in items:
            now = item.t_submit - t0
            # releases due before this arrival are real events at their own
            # deadlines (release-before-place convention): each updates the
            # time-weighted integrals at dl and drains the queue — capacity
            # freed mid-gap can serve waiting items immediately
            for dl, payload in heap.pop_due(now):
                integrate(dl - prev)
                res.empty_node_hours += (dl - prev) * tracker.empty_nodes
                res.unalloc_gpu_hours += (dl - prev) * tracker.unallocated
                # busy = online capacity minus fully-free — under elasticity
                # online shrinks with the pool (static: constant == capacity)
                res.busy_gpu_hours += (dl - prev) * (tracker.online_gpus
                                                     - tracker.fully_free)
                res.billed_node_hours += (dl - prev) * tracker.online_nodes
                res.billed_gpu_hours += (dl - prev) * tracker.online_gpus
                prev = dl
                self._dispatch(payload, dl, res, tracker, heap)
            integrate(now - prev)
            res.empty_node_hours += (now - prev) * tracker.empty_nodes
            res.unalloc_gpu_hours += (now - prev) * tracker.unallocated
            res.busy_gpu_hours += (now - prev) * (tracker.online_gpus
                                                  - tracker.fully_free)
            res.billed_node_hours += (now - prev) * tracker.online_nodes
            res.billed_gpu_hours += (now - prev) * tracker.online_gpus
            prev = now

            q = _Queued(item=item, cls=item.task_class(), k=item.gang_size,
                        t_submit=now, duration=item.duration())
            res.items_total += 1
            res.gangs_total += int(item.is_gang)
            res.singletons_total += int(not item.is_gang)
            # ArrivalObserver hook: the arrival is observed before its own
            # decide call (a trailing estimator legitimately sees it)
            observe = getattr(self.policy, "observe_arrival", None)
            if observe is not None:
                observe(item, now)
            blocked = self.queue_mode == "blocking" and len(self._queue) > 0
            act = None if blocked else self._try_place(q, res, now, tracker)
            if act is None:
                self._enqueue(q, res, now, tracker, heap)
                snap("arr", {"placed": False, "gang": item.is_gang})
            else:
                self._admit(q, act, res, now, heap, tracker)
                snap("arr", {"placed": True, "gang": item.is_gang})

        # drain the tail: remaining releases + final queue state
        t_window_end = now  # last arrival — open streaks clip here so the
        # post-workload drain tail doesn't inflate "continuously empty" stats
        t_end = now
        while len(heap):
            dl, payload = heap.pop()
            t_end = max(t_end, dl)
            now = dl
            integrate(now - prev)
            res.empty_node_hours_total += (now - prev) * tracker.empty_nodes
            res.unalloc_gpu_hours_total += (now - prev) * tracker.unallocated
            res.busy_gpu_hours_total += (now - prev) * (tracker.online_gpus
                                                        - tracker.fully_free)
            res.billed_node_hours_total += (now - prev) * tracker.online_nodes
            res.billed_gpu_hours_total += (now - prev) * tracker.online_gpus
            prev = now
            self._dispatch(payload, now, res, tracker, heap)

        tracker.close_streaks(t_window_end)
        tracker.assert_consistent()
        res.node_max_empty_streak = tracker.max_empty_streak
        res.nodes_never_used = tracker.nodes_never_used
        res.empty_node_hours_total += res.empty_node_hours
        res.unalloc_gpu_hours_total += res.unalloc_gpu_hours
        res.busy_gpu_hours_total += res.busy_gpu_hours
        res.billed_node_hours_total += res.billed_node_hours
        res.billed_gpu_hours_total += res.billed_gpu_hours
        res.arrival_span = t_window_end
        res.span = max(t_end - 0.0, 1e-9)
        res.active_at_end = self._active
        if self._elastic is not None:
            res.launches = self._elastic.launches
            res.evictions = self._elastic.evictions
            res.eviction_regret = self._elastic.eviction_regret
        res.queue_at_end = len(self._queue)
        res.queued_never_placed = len(self._queue)
        res.queued_waits_end = [t_end - q.t_submit for q in self._queue]
        snap("end")
        return res
