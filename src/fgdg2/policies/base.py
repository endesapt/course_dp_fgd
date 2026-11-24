"""Policy protocols — typed contracts replacing duck-typed hooks.

Two placement interfaces (never conflated):
- `SinglePolicy.decide(cluster, cls) -> (node, gpu_idx)` — one task;
- `GangPolicy.decide_gang(cluster, worker_cls, k) -> GangPlan | None` — a
  pure-data plan; the runner owns the two-phase commit (plan/commit split).

Two observer interfaces, renamed to kill the old signature conflict where
one `observe` meant different things (audit finding):
- `PlacementObserver.observe_placement(node_id, cls, placed)` — Layer-1
  lifecycle hook (placement commit AND release), used by ClusteringPolicy;
- `ArrivalObserver.observe_arrival(item, now)` — Layer-2 per-arrival hook
  for tracking M estimators (DynamicMPolicy forwards to its provider);
- `TimeAware.set_time(now)` / `QueueAware.set_queue(items)` — driven by the
  churn runner before every decide.
"""

from typing import Protocol, runtime_checkable

from ..core.cluster import Cluster
from ..core.gang import GangPlan
from ..core.node import Node
from ..core.tasks import TaskClass
from ..items import StreamItem


@runtime_checkable
class SinglePolicy(Protocol):
    def decide(self, cluster: Cluster, cls: TaskClass) -> tuple[Node | None, int | None]: ...


@runtime_checkable
class GangPolicy(Protocol):
    """Gang placement: returns a pure-data plan (the runner commits it).
    `packed=True` means all workers land on ONE node (packed semantics —
    spread otherwise)."""

    packed: bool

    def decide_gang(self, cluster: Cluster, worker_cls: TaskClass, k: int) -> GangPlan | None: ...


@runtime_checkable
class PlacementObserver(Protocol):
    """Layer-1: notified after each placement commit (placed=True) and each
    release (placed=False) — e.g. ClusteringPolicy's hosted-classes memory."""

    def observe_placement(self, node_id: str, cls: TaskClass, placed: bool) -> None: ...


@runtime_checkable
class ArrivalObserver(Protocol):
    """Layer-2: notified once per arrival BEFORE its decide call — a tracking
    estimator legitimately sees the item being decided."""

    def observe_arrival(self, item: StreamItem, now: float) -> None: ...


@runtime_checkable
class TimeAware(Protocol):
    """Driven by the churn runner before every decide — tracking M estimators
    advance their slice here (queue retries score with retry-time M)."""

    def set_time(self, now: float) -> None: ...


@runtime_checkable
class QueueAware(Protocol):
    """Driven by the churn runner before every decide with the current FCFS
    queue snapshot (queue-aware M arms)."""

    def set_queue(self, queue_items, now: float = 0.0) -> None: ...


@runtime_checkable
class RunningAware(Protocol):
    """Driven by the churn runner before every decide with the running-set
    snapshot — (t_start, item) pairs for jobs currently holding resources
    (running-occupancy M arms)."""

    def set_running(self, now: float, entries: list) -> None: ...
