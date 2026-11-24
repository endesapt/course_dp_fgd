"""Elastic node pool — Stage-1 controller (evict-idle>T + reactive scale-up).

The pool owns the node supply (blueprint Nodes minted into fresh incarnations
— ids are NEVER recycled: a live release resolves receipts by id, and a
recycled id would hit the new node object and corrupt state). Nodes launch
with latency delta: a `node_ready` event arrives delta seconds after launch;
billing is per-second from ready (launch latency costs wait, not money).

Two event payloads ride the same EventHeap as job releases:
- _NodeReady — a launched node joins the online pool (then the queue drains).
- _EvictCheck — revalidation token: fires only if the node is STILL empty and
  its empty-clock still matches `since` (a re-emptied node has a fresh clock,
  so stale checks no-op — same-timestamp ordering becomes harmless).

Scale-down is empty-node-only (no migration — the trace has no checkpoint
semantics). Scale-up is shape-aware at the level of the failing item:
a d=8 worker needs a wholly empty node; a k-gang of d-workers needs
k * max(1, d // G) fresh nodes minimum.
"""

import math
from dataclasses import dataclass

from ..core.node import Node


@dataclass
class _NodeReady:
    node: Node


@dataclass
class _EvictCheck:
    node_id: str
    since: float  # the empty_since value at scheduling time (staleness token)


@dataclass
class ElasticConfig:
    """Pool knobs. Defaults: 1h timeout, 300s launch latency, full CSV pool,
    cold start (empty pool — the honest elastic story), no hot reserve."""
    evict_after_s: float = 3600.0
    launch_latency_s: float = 300.0
    max_nodes: int = 0           # 0 = len(blueprints)
    cold_start: bool = True
    reserve_empty: int = 0       # keep-R-hot: never evict below R empty nodes
    reserve_top_n: int = 0       # M1: pin empty nodes for top-N queue gangs
    pin_max_hold_s: float = 21600.0  # pin timeout — release stuck reservations;
    # the expired gang is embargoed from re-pinning for one hold period
    relaunch_cooldown_s: float = 3600.0  # per-item relaunch backoff — without
    # it a gang that can't assemble spins a ready->fail->relaunch heartbeat
    # every delta forever (the drain tail never empties)
    provision_binding: bool = False  # Kueue ProvisioningRequest emulation:
    # nodes launched FOR an item arrive BOUND to it (masked like pins) so
    # provisioned capacity can't be stolen between node_ready and the next
    # drain; a bound booking expires after pin_max_hold_s (BookingExpired
    # analog). Off = unbound launches join the free pool for anyone.


class ElasticPool:
    """Node supply + controller decisions for one churn run."""

    def __init__(self, cfg: ElasticConfig, blueprints: list[Node]):
        if not blueprints:
            raise ValueError("elastic pool needs at least one node blueprint")
        self.cfg = cfg
        self._blueprints = blueprints
        self._rr = 0             # round-robin blueprint pick
        self._incarnation = 0
        self.pending: list[Node] = []   # launched, not yet ready
        self.launches = 0
        self.evictions = 0
        self.eviction_regret = 0  # launches within ~delta of an eviction
        self._last_evict_t: float | None = None
        # M1 reservation: pinned nodes are masked from ordinary placement so
        # a multi-node gang can ASSEMBLE its substrate instead of having the
        # trickle eaten one node at a time by smaller queue items
        self.reserved_nids: set[str] = set()
        self.pins: dict[str, set[str]] = {}      # gang_id -> pinned node ids
        self._pin_since: dict[str, float] = {}   # gang_id -> first pin time
        self._pin_embargo: dict[str, float] = {}  # gang_id -> re-pin banned till
        self.pin_events = 0
        self._last_launch_for: dict[str, float] = {}  # relaunch backoff
        self._bound_pending: dict[str, str] = {}  # node_id -> gang_id

    # -- supply ---------------------------------------------------------------
    def _mint(self) -> Node:
        bp = self._blueprints[self._rr % len(self._blueprints)]
        self._rr += 1
        self._incarnation += 1
        return Node(f"{bp.id}#i{self._incarnation}", bp.num_gpus,
                    model=bp.model, cpu_milli=bp.cpu_milli,
                    memory_mib=bp.memory_mib)

    @property
    def max_nodes(self) -> int:
        return self.cfg.max_nodes or len(self._blueprints)

    def online_plus_pending(self, tracker) -> int:
        return tracker.online_nodes + len(self.pending)

    def min_fresh_nodes(self, cls, k: int) -> int | None:
        """Nodes needed for this item alone on fresh empty nodes: workers
        pack floor(G/d) per node. None = never feasible in this pool (worker
        too big or gang wider than the pool) — launching cannot help it."""
        g = max(n.num_gpus for n in self._blueprints)
        cpu = max(n.cpu_milli for n in self._blueprints)
        mem = max(n.memory_mib for n in self._blueprints)
        d = cls.gpu_demand
        if d > g or cls.cpu_milli > cpu or cls.memory_mib > mem:
            return None
        need = math.ceil(k / max(1, int(g // d)))
        return need if need <= self.max_nodes else None

    # -- scale-up (reactive) ----------------------------------------------------
    def on_enqueue(self, gang_id, cls, k: int, now: float, tracker,
                   heap) -> int:
        """An item failed placement and queued. Reactive trigger: if its
        fresh-node need isn't already covered by launching nodes or its own
        pins, launch the gap — bounded by the pool cap. Returns launches."""
        need = self.min_fresh_nodes(cls, k)
        if need is None:
            return 0  # never feasible in this pool — no launch helps
        if (now - self._last_launch_for.get(gang_id, -1e18)
                < self.cfg.relaunch_cooldown_s):
            return 0  # already launched for this item recently — back off
        # coverage accounting: unbound mode counts ALL in-flight launches
        # (they'll join the free pool); bound mode counts only this item's
        # own provisioned nodes — capacity bound to another item is spoken
        # for, so each queued item requests its own (Kueue's per-Workload
        # ProvisioningRequest semantics)
        covered = (sum(1 for g in self._bound_pending.values()
                      if g == gang_id)
                   if self.cfg.provision_binding else len(self.pending))
        gap = need - covered - len(self.pins.get(gang_id, ()))
        if gap <= 0:
            return 0
        room = self.max_nodes - self.online_plus_pending(tracker)
        n = min(gap, room)
        for _ in range(max(0, n)):
            node = self._mint()
            self.pending.append(node)
            if self.cfg.provision_binding:
                self._bound_pending[node.id] = gang_id
            heap.push(now + self.cfg.launch_latency_s, _NodeReady(node))
            self.launches += 1
        if n > 0:
            self._last_launch_for[gang_id] = now
        if n > 0 and self._last_evict_t is not None and (
                now - self._last_evict_t) <= self.cfg.launch_latency_s:
            self.eviction_regret += n
        return max(0, n)

    def node_ready(self, payload: _NodeReady, now: float, tracker,
                   queue=None) -> bool:
        if payload.node not in self.pending:
            return False  # stale ready event (double-fired) — ignore
        self.pending.remove(payload.node)
        tracker.add(payload.node, now)
        gid = self._bound_pending.pop(payload.node.id, None)
        if (gid is not None and queue is not None
                and any(q.item.gang_id == gid for q in queue)):
            # the ProvisioningRequest lands: bind to its item via the pin
            # mask so nobody can steal it before the next drain. An item
            # that already placed (or never existed) leaves the node free.
            self.pins.setdefault(gid, set()).add(payload.node.id)
            self.reserved_nids.add(payload.node.id)
            self._pin_since.setdefault(gid, now)
            self.pin_events += 1
        return True

    # -- scale-down --------------------------------------------------------------
    def on_emptied(self, nids: list, now: float, heap) -> None:
        """Nodes that just became fully empty — arm an eviction check."""
        t = now + self.cfg.evict_after_s
        for nid in nids:
            heap.push(t, _EvictCheck(nid, now))

    def arm_initial(self, tracker, heap) -> None:
        """Warm start: schedule checks for nodes already empty at t=0."""
        for nid, since in tracker.empty_since.items():
            heap.push(since + self.cfg.evict_after_s, _EvictCheck(nid, since))

    def evict_check(self, payload: _EvictCheck, now: float,
                    tracker) -> bool:
        """Fire iff the node is still online, still empty, and its idle clock
        is the one this check was armed with — plus the hot-reserve floor.
        Pinned nodes are exempt: reservation owns their idle clock."""
        nid = payload.node_id
        if tracker.empty_since.get(nid) != payload.since:
            return False  # stale (re-emptied or evicted already)
        if nid in self.reserved_nids:
            return False
        if tracker.empty_nodes <= self.cfg.reserve_empty:
            return False  # reserve floor reached — node stays as hot substrate
        tracker.evict(nid, now)
        self.evictions += 1
        self._last_evict_t = now
        return True

    # -- M1 top-N reservation ----------------------------------------------------
    def assign_pins(self, queue, now: float, tracker, heap) -> None:
        """Pin empty online nodes for the first `reserve_top_n` queue items
        that need >=2 fresh nodes — gangs assemble substrate instead of
        having the trickle eaten by smaller queue items (the starvation fix).
        Called whenever availability changes: enqueue, release, node_ready.
        Expired pins (held > pin_max_hold_s without the gang placing) drop AND
        the gang is embargoed from re-pinning for one hold period — without the
        embargo an expired top-N gang re-pins in this same call, which makes
        pin_max_hold_s a no-op and lets masked capacity be held forever."""
        # expiry sweep runs for EVERY pin source — top-N reservations AND
        # bound provisioned nodes (their expiry is the BookingExpired
        # analog) — so it precedes the reserve_top_n early-return
        for gid, t0 in list(self._pin_since.items()):
            if now - t0 >= self.cfg.pin_max_hold_s:
                self.release_pins(gid, now, tracker, heap)
                self._pin_embargo[gid] = now + self.cfg.pin_max_hold_s
        if self.cfg.reserve_top_n <= 0:
            return
        needers = 0
        for q in queue:
            gid = q.item.gang_id
            if self._pin_embargo.get(gid, 0.0) > now:
                continue  # embargoed: can't pin and doesn't count against N
            quota = self.min_fresh_nodes(q.cls, q.k)
            if quota is None or quota < 2:
                continue  # infeasible or singleton-ish — pins don't help it
            needers += 1
            if needers > self.cfg.reserve_top_n:
                break
            have = self.pins.setdefault(gid, set())
            if len(have) >= quota:
                continue
            self._pin_since.setdefault(gid, now)
            for nid in list(tracker.empty_since.keys()):
                if len(have) >= quota:
                    break
                if nid in self.reserved_nids:
                    continue
                have.add(nid)
                self.reserved_nids.add(nid)
                self.pin_events += 1
            if not have:
                self.pins.pop(gid, None)
                self._pin_since.pop(gid, None)

    def release_pins(self, gang_id, now: float, tracker, heap) -> None:
        """Drop a gang's pins — it placed (on them or elsewhere) or timed
        out. A still-empty freed node kept its ORIGINAL idle clock the whole
        time it was pinned (pinning never touched empty_since), so its check
        is re-armed with that same token at empty_since+T — firing now if the
        deadline already passed while pinned. Re-arming with `now` instead
        would stamp a token that never matches empty_since: the check fires
        stale, nothing re-arms, and the node stays billed-empty forever."""
        pinned = self.pins.pop(gang_id, None)
        self._pin_since.pop(gang_id, None)
        self._pin_embargo.pop(gang_id, None)
        if not pinned:
            return
        for nid in pinned:
            self.reserved_nids.discard(nid)
            since = tracker.empty_since.get(nid)
            if since is not None:
                heap.push(max(since + self.cfg.evict_after_s, now),
                          _EvictCheck(nid, since))
