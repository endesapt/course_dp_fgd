"""Tracking M estimators (Phase-3 arms, direction study 2026-09-22).

Each provider emits the class histogram a different way; all share the
semantics of the old SlidingMPolicy modes:

- "lead"  — M_t = class histogram of items with submit_time in [t, t+W]:
      the per-slice FUTURE oracle (perfect foresight at granularity W;
      strictly >= the fixed window-oracle, which averages within-window drift).
- "trail" — M_t = histogram of arrivals in (t-W, t]: the windowed empirical
      estimator a deployment could actually run.
- "ewma"  — M_t = EWMA over observed arrivals, p <- (1-eta)*p + eta*e_m,
      seeded with a PRIOR workload (e.g. last week's stale M). Support never
      collapses — unseen classes decay smoothly instead of dropping out
      (what killed the trail arms).
- "trail_occ" — like trail, but each arrival is weighted by its class's
      EXPECTED duration (λ_m·E[L_m]) instead of counting once: the
      deployable occupancy estimator. Duration means come from a PAST
      window's table; weights=None uses each item's own duration_s
      (runtime-estimate variant — semi-oracular).
- "queue" — M_t = alpha*(histogram of items currently in the FCFS queue)
      + (1-alpha)*prior. The deployable estimator needing NO prediction:
      under congestion the decision-relevant mix is what is waiting to be
      placed — which the scheduler observes directly.
- "running" — M_t = alpha*(histogram of the RUNNING set) + (1-alpha)*prior.
      The observable real-time version of 'occupancy': a scheduler always
      knows what currently holds its resources, no prediction needed.
      Weight modes: 'held' = elapsed time held (now - t_start, the
      λ_m·E[L_m] analog), 'count' = one vote per running job, 'dur' =
      full duration_s (runtime-estimate variant — semi-oracular).

Shared semantics: the histogram is per-ITEM (job level — a gang of k counts
once, same as empirical_workload); the class key is (worker_cpu_milli,
worker_gpu_demand). When a slice is empty (long gaps, the post-arrival drain
tail) the last non-empty histogram is kept.
"""

import math
from collections import Counter, deque

from ..core.tasks import workload_from_counts
from .base import TrackingM


class LeadWindowM(TrackingM):
    """Per-slice future oracle: histogram of items submitting in [t, t+W]."""

    mode = "lead"

    def __init__(self, items, window_s: float, oracle=None):
        super().__init__(oracle)
        if window_s is None or window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.window_s = float(window_s)
        # two pointers over the submit-sorted item list; times normalized by
        # the first submit to match the runner's `now` origin
        self._items = sorted(items, key=lambda it: it.t_submit)
        t0 = self._items[0].t_submit if self._items else 0.0
        self._times = [it.t_submit - t0 for it in self._items]
        self._lo = self._hi = 0
        self._counts: Counter = Counter()

    def set_time(self, now: float) -> None:
        changed = False
        items, times = self._items, self._times
        while self._lo < len(items) and times[self._lo] < now:
            self._counts[items[self._lo].class_key] -= 1
            self._lo += 1
            changed = True
        while self._hi < len(items) and times[self._hi] < now + self.window_s:
            self._counts[items[self._hi].class_key] += 1
            self._hi += 1
            changed = True
        if not changed and self.workload is not None:
            return
        active = {k: c for k, c in self._counts.items() if c > 0}
        if not active:
            return  # keep the last non-empty histogram (drain tail, gaps)
        self._emit(workload_from_counts(active))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "slide_window_s": self.window_s}


class TrailWindowM(TrackingM):
    """Windowed empirical estimator: histogram of arrivals in (t-W, t]."""

    mode = "trail"

    def __init__(self, window_s: float, oracle=None):
        super().__init__(oracle)
        if window_s is None or window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.window_s = float(window_s)
        self._seen: deque = deque()
        self._counts: Counter = Counter()

    def observe_arrival(self, item, now: float) -> None:
        self._seen.append((now, item.class_key))
        self._counts[item.class_key] += 1

    def set_time(self, now: float) -> None:
        changed = False
        cut = now - self.window_s
        while self._seen and self._seen[0][0] <= cut:
            _, k = self._seen.popleft()
            self._counts[k] -= 1
            changed = True
        # the CURRENT item arrives via observe_arrival() before set_time, so it
        # is already inside the window — intentional and realistic (the
        # estimator has seen the item being decided).
        if not changed and self.workload is not None:
            return
        active = {k: c for k, c in self._counts.items() if c > 0}
        if not active:
            return
        self._emit(workload_from_counts(active))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "slide_window_s": self.window_s}


class TrailOccupancyM(TrackingM):
    """Trailing-window occupancy estimator: arrivals in (t-W, t] weighted by
    expected class duration rather than counted once — λ_m·E[L_m] realized
    on the trailing window.

    `weights` = {class_key: E[duration_s]} from a past window (deployable);
    `weights=None` uses each item's own duration_s (the runtime-estimate
    variant — assumes accurate submit-time durations)."""

    mode = "trail_occ"

    def __init__(self, window_s, weights=None, default_w=None, oracle=None):
        super().__init__(oracle)
        if window_s is None or window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.window_s = float(window_s)
        self._weights = weights
        self._default_w = default_w
        self._seen: deque = deque()  # (t, class_key, weight)
        self._counts: Counter = Counter()

    def _w(self, item) -> float:
        if self._weights is None:
            if item.duration_s is None:
                raise ValueError(
                    "trail_occ with self weights needs item duration_s "
                    "(durations=field or a duration table)"
                )
            return float(item.duration_s)
        w = self._weights.get(item.class_key, self._default_w)
        if w is None:
            raise ValueError(f"trail_occ: no duration weight for {item.class_key}")
        return float(w)

    def observe_arrival(self, item, now: float) -> None:
        w = self._w(item)
        self._seen.append((now, item.class_key, w))
        self._counts[item.class_key] += w

    def set_time(self, now: float) -> None:
        changed = False
        cut = now - self.window_s
        while self._seen and self._seen[0][0] <= cut:
            _, k, w = self._seen.popleft()
            self._counts[k] -= w
            changed = True
        if not changed and self.workload is not None:
            return
        active = {k: c for k, c in self._counts.items() if c > 0}
        if not active:
            return
        self._emit(workload_from_counts(active))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "slide_window_s": self.window_s}


class EwmaM(TrackingM):
    """EWMA over observed arrivals seeded by a prior workload."""

    mode = "ewma"

    def __init__(self, eta: float, prior, oracle=None):
        super().__init__(oracle)
        if eta is None or not 0 < eta <= 1:
            raise ValueError(f"ewma eta must be in (0, 1], got {eta}")
        if prior is None:
            raise ValueError("ewma mode requires a seed prior Workload")
        self.eta = float(eta)
        self._p = {
            (c.cpu_milli, c.gpu_demand): float(w)
            for c, w in zip(prior.classes, prior.popularities)
        }
        self._dirty = False

    def observe_arrival(self, item, now: float) -> None:
        k = item.class_key
        for kk in self._p:
            self._p[kk] *= 1.0 - self.eta
        self._p[k] = self._p.get(k, 0.0) + self.eta
        self._dirty = True

    def set_time(self, now: float) -> None:
        if not self._dirty and self.workload is not None:
            return
        self._dirty = False
        self._emit(workload_from_counts(self._p))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "slide_eta": self.eta}


class QueueAwareM(TrackingM):
    """M_t = alpha * queue-histogram + (1 - alpha) * prior — the deployable
    congestion-aware estimator (no prediction needed).

    set_queue receives _Queued wrappers (with .class_key and relative
    .t_submit); plain StreamItems are tolerated for tests/back-compat."""

    mode = "queue"

    def __init__(self, alpha: float, prior, oracle=None):
        super().__init__(oracle)
        if alpha is None or not 0 < alpha <= 1:
            raise ValueError(f"queue alpha must be in (0, 1], got {alpha}")
        if prior is None:
            raise ValueError("queue mode requires a prior Workload")
        self.alpha = float(alpha)
        self._prior_p = {
            (c.cpu_milli, c.gpu_demand): float(w)
            for c, w in zip(prior.classes, prior.popularities)
        }
        self._last_w = None

    def _queue_weights(self, queue_items, now: float):
        """class_key -> raw vote. Base: every queued item votes once."""
        return Counter(it.class_key for it in queue_items)

    def set_queue(self, queue_items, now: float = 0.0) -> None:
        counts = self._queue_weights(queue_items, now)
        tot = sum(counts.values())
        w = {k: self.alpha * c / tot for k, c in counts.items()} if tot else {}
        for k, p in self._prior_p.items():
            w[k] = w.get(k, 0.0) + (1.0 - self.alpha) * p
        if not any(v > 0 for v in w.values()):
            w = dict(self._prior_p)  # empty queue at alpha=1 -> pure prior
        w = {k: v for k, v in w.items() if v > 0}
        if w == self._last_w:
            return
        self._last_w = w
        self._emit(workload_from_counts(w))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "queue_alpha": self.alpha}


class QueueAgeM(QueueAwareM):
    """Queue-M with exponential age decay: a queued item votes
    exp(-(now - t_submit)/tau). Stuck-for-days items (the backlog
    'cemetery' — classes that provably cannot place) decay toward zero
    instead of dominating M; the estimate tracks the *fresh*, actionable
    backlog. tau=inf recovers plain QueueAwareM."""

    mode = "queueage"

    def __init__(self, tau: float, alpha: float, prior, oracle=None):
        super().__init__(alpha, prior, oracle=oracle)
        if tau is None or not tau > 0:
            raise ValueError(f"queue_age tau must be > 0 s, got {tau}")
        self.tau = float(tau)

    def _queue_weights(self, queue_items, now: float):
        w = Counter()
        for it in queue_items:
            t = getattr(it, "t_submit", None)
            age = max(now - t, 0.0) if t is not None else 0.0
            w[it.class_key] += math.exp(-age / self.tau)
        return w

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "queue_tau_s": self.tau}


class QueueCapM(QueueAwareM):
    """Queue-M restricted to placeable classes: a queued item votes only if
    its total GPU demand <= cap. Giant gangs that exceed the cap (the ones
    that sit for days because no allocation can serve them) are invisible
    to M — the estimator learns from the backlog it could actually clear."""

    mode = "queuecap"

    def __init__(self, cap: float, alpha: float, prior, oracle=None):
        super().__init__(alpha, prior, oracle=oracle)
        if cap is None or not cap > 0:
            raise ValueError(f"queuecap demand cap must be > 0, got {cap}")
        self.cap = float(cap)

    def _queue_weights(self, queue_items, now: float):
        w = Counter()
        for it in queue_items:
            item = getattr(it, "item", it)
            if item.gpu_demand_total <= self.cap:
                w[it.class_key] += 1
        return w

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "queue_cap": self.cap}


class RunningOccupancyM(TrackingM):
    """M_t = alpha * (running-set histogram) + (1 - alpha) * prior — the
    observable version of the 'occupancy' arm: instead of weighting arrivals
    by true durations (semi-oracular) or a past window (stale), weight the
    jobs currently holding the cluster, which the scheduler sees directly.

    `weight`: 'held' = time held so far (now - t_start); 'count' = each
    running job votes once; 'dur' = item's full duration_s (needs duration
    estimates — the same semi-oracular caveat as trail_occ:self)."""

    mode = "running"

    def __init__(self, weight: str, alpha: float, prior, oracle=None):
        super().__init__(oracle)
        if weight not in ("held", "count", "dur"):
            raise ValueError(
                f"running weight must be held|count|dur, got {weight!r}")
        if alpha is None or not 0 < alpha <= 1:
            raise ValueError(f"running alpha must be in (0, 1], got {alpha}")
        if prior is None:
            raise ValueError("running mode requires a prior Workload")
        self.weight = weight
        self.alpha = float(alpha)
        self._prior_p = {
            (c.cpu_milli, c.gpu_demand): float(w)
            for c, w in zip(prior.classes, prior.popularities)
        }
        self._last_w = None

    def set_running(self, now: float, entries) -> None:
        counts: Counter = Counter()
        for t_start, item in entries:
            if self.weight == "held":
                w = max(now - t_start, 1e-9)
            elif self.weight == "dur":
                w = float(item.duration())  # raises clearly if duration_s absent
            else:
                w = 1.0
            counts[item.class_key] += w
        tot = sum(counts.values())
        w = {k: self.alpha * c / tot for k, c in counts.items()} if tot else {}
        for k, p in self._prior_p.items():
            w[k] = w.get(k, 0.0) + (1.0 - self.alpha) * p
        if not any(v > 0 for v in w.values()):
            w = dict(self._prior_p)  # empty running set at alpha=1 -> prior
        w = {k: v for k, v in w.items() if v > 0}
        if w == self._last_w:
            return
        self._last_w = w
        self._emit(workload_from_counts(w))

    def tracking_stats(self) -> dict:
        return {**super().tracking_stats(), "run_weight": self.weight,
                "queue_alpha": self.alpha}


class CompositeM(TrackingM):
    """Convex mixture over sub-estimators: M_t = sum_i w_i * M_i(t).

    Static terms contribute their fixed workload; tracking terms re-emit on
    change. Every hook is forwarded to all tracking terms; the mixture is
    re-emitted whenever any term's workload object was reassigned. Terms with
    an internal prior (queue/running/ewma) keep their empty-state fallback —
    inside a mix that fallback is the sub-prior, so weight terms compose
    cleanly (e.g. queue:1:<P> contributes the queue signal, falling back to
    P only while the queue is empty)."""

    mode = "mix"

    def __init__(self, terms, oracle=None):
        """terms: [(weight, provider|None, static_workload)] — provider is
        None for static specs, which hold `static_workload` permanently."""
        super().__init__(oracle)
        self._terms = [(float(w), prov, wl) for w, prov, wl in terms]
        self._seen = [None] * len(self._terms)

    def _forward(self, method, *args):
        for _, prov, _ in self._terms:
            if prov is not None:
                getattr(prov, method)(*args)
        self._maybe_emit()

    def observe_arrival(self, item, now: float) -> None:
        self._forward("observe_arrival", item, now)

    def set_time(self, now: float) -> None:
        self._forward("set_time", now)

    def set_queue(self, queue_items, now: float = 0.0) -> None:
        self._forward("set_queue", queue_items, now)

    def set_running(self, now: float, entries) -> None:
        self._forward("set_running", now, entries)

    def _maybe_emit(self) -> None:
        changed = False
        for i, (_, prov, wl) in enumerate(self._terms):
            cur = prov.workload if prov is not None else wl
            if cur is not self._seen[i]:
                self._seen[i] = cur
                changed = True
        # wait until every tracking term has produced its first estimate
        if not changed or any(w is None for w in self._seen):
            return
        mix: Counter = Counter()
        for (w, _, _), wl in zip(self._terms, self._seen):
            for c, p in zip(wl.classes, wl.popularities):
                mix[(c.cpu_milli, c.gpu_demand)] += w * float(p)
        self._emit(workload_from_counts(mix))


TRACKING_MODES = {
    "lead": LeadWindowM,
    "trail": TrailWindowM,
    "trail_occ": TrailOccupancyM,
    "ewma": EwmaM,
    "queue": QueueAwareM,
    "running": RunningOccupancyM,
    "mix": CompositeM,
}
