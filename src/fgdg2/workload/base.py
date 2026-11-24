"""M-provider protocols — who supplies the workload the policy scores against.

The FGD objective needs a class distribution M. Architecturally that is an
ESTIMATOR with its own inputs, not a policy field:

- static specs (stream/empirical/occupancy/tilted/stale) resolve to one fixed
  `Workload` — no provider object, the policy just stores it;
- tracking specs (lead/trail/ewma/queue) resolve to a `TrackingM` provider that
  the runner feeds through the DynamicMPolicy adapter's hooks:
  `observe_arrival` (per arrival), `set_time` (before every decide),
  `set_queue` (before every decide, FCFS snapshot).

`TrackingM.workload` is only REASSIGNED when the estimate actually changed —
the wrapper rebuilds its inner policy on identity change, so estimator
churn and policy rebuilds stay in lockstep.
"""

from ..core.tasks import Workload
from ..items import StreamItem
from .construct import workload_dtv


class TrackingM:
    """Base class for time/queue-driven M estimators.

    Subclasses override the hooks they use. `rebuilds`/`dtv_sum` are bumped
    only on actual estimate changes (via `_emit`), so `tracking_stats()["…rebuilds"]`
    counts distinct M_t values over the run.
    """

    #: set by subclasses to label the arm in artifacts/stats
    mode: str = "base"

    def __init__(self, oracle: Workload | None = None):
        self.oracle = oracle
        self.workload: Workload | None = None
        self.rebuilds = 0
        self.dtv_sum = 0.0

    # -- runner-facing hooks (all no-ops unless overridden) -----------------
    def observe_arrival(self, item: StreamItem, now: float) -> None:
        """Called once per arrival, before its own decide (the estimator may
        legitimately see the item being decided)."""

    def set_time(self, now: float) -> None:
        """Advance to sim time `now` (queue retries score with retry-time M)."""

    def set_queue(self, queue_items, now: float = 0.0) -> None:
        """Snapshot of queued items right before a decide (queue-aware arms).
        Entries are the runner's queued wrappers exposing .class_key and
        relative .t_submit (plain StreamItems also accepted)."""

    def set_running(self, now: float, entries) -> None:
        """Running-set snapshot right before a decide: (t_start, item) pairs
        of jobs currently holding resources (running-occupancy arms)."""

    # -- internals -----------------------------------------------------------
    def _emit(self, wl: Workload) -> None:
        """Adopt a new estimate: rebuild counter + d_TV bookkeeping vs oracle."""
        self.workload = wl
        self.rebuilds += 1
        if self.oracle is not None:
            self.dtv_sum += workload_dtv(wl, self.oracle)

    def tracking_stats(self) -> dict:
        return {
            "slide_mode": self.mode,
            "slide_rebuilds": self.rebuilds,
            "slide_mean_dtv_vs_oracle": (
                self.dtv_sum / self.rebuilds if self.rebuilds else None
            ),
        }
