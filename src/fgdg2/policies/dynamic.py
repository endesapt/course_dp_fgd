"""DynamicMPolicy — the adapter between a tracking M provider and the
gang/background policy pair.

Replaces the old SlidingMPolicy (which mixed estimation + policy plumbing).
Here the ESTIMATOR lives in `workload/tracking.py` (a TrackingM provider owns
the histogram math and decides when M changed); this wrapper only rebuilds
the inner policy pair when `provider.workload` is reassigned.

`build_inner(workload)` is called on every rebuild — must be cheap
(DPFGDGang/SequentialGangPolicy just store the workload).

Hook order per decide (driven by the churn runner): observe_arrival (once per
arrival) -> set_time -> set_queue -> decide_gang. Queue retries therefore
score with the retry-time M.
"""

from ..workload.base import TrackingM


class DynamicMPolicy:
    """Wraps `build_inner` + a TrackingM provider into a GangPolicy whose M
    tracks the stream."""

    def __init__(self, build_inner, provider: TrackingM):
        self._build_inner = build_inner
        self.provider = provider
        self._inner = None
        self._applied = None  # the workload object the inner pair was built on

    # -- runner hooks (ArrivalObserver/TimeAware/QueueAware protocols) -------
    def _refresh(self) -> None:
        wl = self.provider.workload
        if wl is not None and wl is not self._applied:
            self._inner = self._build_inner(wl)
            self._applied = wl

    def observe_arrival(self, item, now: float) -> None:
        self.provider.observe_arrival(item, now)
        self._refresh()

    def set_time(self, now: float) -> None:
        self.provider.set_time(now)
        self._refresh()

    def set_queue(self, queue_items, now: float = 0.0) -> None:
        self.provider.set_queue(queue_items, now)
        self._refresh()

    def set_running(self, now: float, entries) -> None:
        self.provider.set_running(now, entries)
        self._refresh()

    # -- policy interface ----------------------------------------------------
    def decide_gang(self, cluster, worker_cls, k: int):
        if self._inner is None:
            raise RuntimeError(
                "DynamicMPolicy used before its provider produced an estimate "
                "(call set_time/set_queue first — the churn runner does)"
            )
        return self._inner.decide_gang(cluster, worker_cls, k)

    @property
    def packed(self) -> bool:
        return bool(getattr(self._inner, "packed", False))

    def tracking_stats(self) -> dict:
        return self.provider.tracking_stats()
