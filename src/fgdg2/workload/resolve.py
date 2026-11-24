"""Resolve an MSpec against a concrete stream → (Workload, TrackingM|None).

`StreamContext` bundles what resolution needs: the items, the stream header
(config drives 'stream'), the base M_whole/M_frac the stream was generated
from (fingerprint-checked upstream), and the window oracle (the empirical
histogram — computed lazily so 'stream' doesn't pay for it).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..core.tasks import Workload, workload_from_counts
from ..datasets.durations import duration_means
from ..datasets.synth import policy_workload
from ..items import StreamItem
from .base import TrackingM
from .construct import (
    empirical_workload,
    empirical_workload_from_file,
    occupancy_workload,
    occupancy_workload_from_file,
    tilted_workload,
)
from .spec import MSpec, parse_mspec
from .tracking import (
    CompositeM,
    EwmaM,
    LeadWindowM,
    QueueAgeM,
    QueueAwareM,
    QueueCapM,
    RunningOccupancyM,
    TrailOccupancyM,
    TrailWindowM,
)


@dataclass
class StreamContext:
    items: list[StreamItem]
    header: dict
    m_whole: Workload
    m_frac: Workload | None = None
    _oracle: Workload | None = field(default=None, repr=False)

    @property
    def oracle(self) -> Workload:
        """Empirical class histogram of the replayed window — the perfect-
        foresight M at window granularity (the 'empirical' arm)."""
        if self._oracle is None:
            self._oracle = empirical_workload(self.items)
        return self._oracle


@dataclass
class MResolution:
    workload: Workload          # static M, or the seed prior for tracking arms
    provider: TrackingM | None  # non-None for lead/trail/ewma/queue
    detail: dict                # extra config fields for the artifact
    slug: str                   # compact tag for filenames/summaries


def resolve_mspec(spec: str | MSpec, ctx: StreamContext) -> MResolution:
    """MSpec → MResolution. Tracking arms return the provider; the `workload`
    field is then only the SEED the provider/policies start from."""
    m = parse_mspec(spec) if isinstance(spec, str) else spec

    if m.kind == "stream":
        return MResolution(
            policy_workload(ctx.m_whole, ctx.header, ctx.m_frac), None, {}, m.slug)
    if m.kind == "empirical":
        return MResolution(ctx.oracle, None, {}, m.slug)
    if m.kind == "occupancy":
        t_end = (ctx.header["config"].get("layer2_window") or {}).get("t_hi")
        return MResolution(
            occupancy_workload(ctx.items, t_end),
            None, {"occupancy_t_end": t_end}, m.slug)
    if m.kind == "tilted":
        base = (resolve_mspec(m.prior, ctx).workload if m.prior is not None
                else ctx.oracle)
        return MResolution(
            tilted_workload(base, m.gamma), None,
            {"gamma": m.gamma, "prior": m.prior.slug if m.prior else "oracle"},
            m.slug)
    if m.kind == "stale":
        return MResolution(
            empirical_workload_from_file(m.source),
            None, {"stale_source": m.source}, m.slug)
    if m.kind == "stale_occ":
        return MResolution(
            occupancy_workload_from_file(m.source),
            None, {"stale_occ_source": m.source}, m.slug)

    # tracking arms — the provider rebuilds M_t during the run; the returned
    # workload is the initial estimate / seed prior
    if m.kind in ("lead", "trail"):
        cls = LeadWindowM if m.kind == "lead" else TrailWindowM
        provider = (
            cls(ctx.items, m.window_s, oracle=ctx.oracle)
            if m.kind == "lead"
            else cls(m.window_s, oracle=ctx.oracle)
        )
        detail = {"slide_mode": m.kind, "slide_window_s": m.window_s}
        return MResolution(ctx.oracle, provider, detail, m.slug)
    if m.kind == "trail_occ":
        # duration-weighted trailing window: 'self' = item's own duration_s
        # (runtime-estimate variant); otherwise a past-window mean table
        if m.source == "self":
            provider = TrailOccupancyM(m.window_s, oracle=ctx.oracle)
        else:
            table = duration_means(m.source)
            provider = TrailOccupancyM(
                m.window_s, weights=table,
                default_w=float(np.mean(list(table.values()))),
                oracle=ctx.oracle)
        detail = {"slide_mode": "trail_occ", "slide_window_s": m.window_s,
                  "occ_dur_source": m.source}
        return MResolution(ctx.oracle, provider, detail, m.slug)
    if m.kind == "ewma":
        prior = resolve_mspec(m.prior, ctx)
        provider = EwmaM(m.eta, prior.workload, oracle=ctx.oracle)
        detail = {"slide_mode": "ewma", "slide_eta": m.eta,
                  "slide_prior": _spec_str(m.prior)}
        return MResolution(prior.workload, provider, detail, m.slug)
    if m.kind == "queue":
        prior = resolve_mspec(m.prior, ctx)
        provider = QueueAwareM(m.alpha, prior.workload, oracle=ctx.oracle)
        detail = {"slide_mode": "queue", "queue_alpha": m.alpha,
                  "slide_prior": _spec_str(m.prior)}
        return MResolution(prior.workload, provider, detail, m.slug)
    if m.kind == "queueage":
        prior = resolve_mspec(m.prior, ctx)
        provider = QueueAgeM(m.tau_s, m.alpha, prior.workload,
                             oracle=ctx.oracle)
        detail = {"slide_mode": "queue_age", "queue_tau_s": m.tau_s,
                  "queue_alpha": m.alpha, "slide_prior": _spec_str(m.prior)}
        return MResolution(prior.workload, provider, detail, m.slug)
    if m.kind == "queuecap":
        prior = resolve_mspec(m.prior, ctx)
        provider = QueueCapM(m.cap, m.alpha, prior.workload,
                             oracle=ctx.oracle)
        detail = {"slide_mode": "queuecap", "queue_cap": m.cap,
                  "queue_alpha": m.alpha, "slide_prior": _spec_str(m.prior)}
        return MResolution(prior.workload, provider, detail, m.slug)
    if m.kind == "running":
        prior = resolve_mspec(m.prior, ctx)
        provider = RunningOccupancyM(m.source, m.alpha, prior.workload,
                                     oracle=ctx.oracle)
        detail = {"slide_mode": "running", "run_weight": m.source,
                  "queue_alpha": m.alpha, "slide_prior": _spec_str(m.prior)}
        return MResolution(prior.workload, provider, detail, m.slug)
    if m.kind == "mix":
        terms = []
        seed_mix: Counter = Counter()
        for w, t in m.terms:
            r = resolve_mspec(t, ctx)
            terms.append((w, r.provider, r.workload))
            for c, p in zip(r.workload.classes, r.workload.popularities):
                seed_mix[(c.cpu_milli, c.gpu_demand)] += w * float(p)
        provider = CompositeM(terms, oracle=ctx.oracle)
        detail = {"slide_mode": "mix",
                  "mix_terms": [f"{w:g}@{_spec_str(t)}" for w, t in m.terms]}
        return MResolution(
            workload_from_counts(seed_mix), provider, detail, m.slug)
    raise ValueError(f"unhandled MSpec kind {m.kind!r}")


def _spec_str(m: MSpec) -> str:
    """Re-render a parsed spec (for detail records) — inverse of parse."""
    if m.kind in ("stream", "empirical", "occupancy"):
        return m.kind
    if m.kind == "tilted":
        return f"tilted:{m.gamma:g}"
    if m.kind == "stale":
        return f"stale:{m.source}"
    if m.kind == "stale_occ":
        return f"stale_occ:{m.source}"
    if m.kind == "trail_occ":
        return f"slide_trail_occ:{m.window_s:g}:{m.source}"
    if m.kind in ("lead", "trail"):
        return f"slide_{m.kind}:{m.window_s:g}"
    if m.kind == "ewma":
        return f"slide_ewma:{m.eta:g}:{_spec_str(m.prior)}"
    if m.kind == "queue":
        return f"queue:{m.alpha:g}:{_spec_str(m.prior)}"
    if m.kind == "queueage":
        return f"queue_age:{m.tau_s:g}:{m.alpha:g}:{_spec_str(m.prior)}"
    if m.kind == "queuecap":
        return f"queuecap:{m.cap:g}:{m.alpha:g}:{_spec_str(m.prior)}"
    if m.kind == "running":
        return f"running:{m.source}:{m.alpha:g}:{_spec_str(m.prior)}"
    if m.kind == "mix":
        return "mix:" + "|".join(f"{w:g}@{_spec_str(t)}" for w, t in m.terms)
    return m.slug
