"""The --policy-m spec DSL — pure parsing, no data dependency.

Specs (Phase-1/3 M-design arms, direction study 2026-09-22 §3):
- 'stream'                  — header-config reconstruction (synthetic streams)
- 'empirical'               — the replayed window's own class histogram =
                              ORACLE at window granularity
- 'occupancy'               — same window but duration-weighted (λ_m·E[L_m])
- 'tilted:<γ>'              — oracle popularities reweighted by d^γ (γ=0 = oracle)
- 'tilted:<γ>:<prior>'      — same tilt applied to a nested static prior
                            (e.g. tilted:1:stale:<file> = deployable demand-
                            weighted estimate: past-window histogram × d)
- 'stale:<file>'            — histogram of a PAST window (.jsonl stream or
                              .csv pods)
- 'slide_lead:<s>'          — sliding M, per-slice future oracle [t, t+s]
- 'slide_trail:<s>'         — sliding M, trailing window (t-s, t]
- 'slide_ewma:<η>:<prior>'  — EWMA estimator seeded by a nested prior spec
- 'queue:<α>:<prior>'       — α·queue-histogram + (1−α)·prior
- 'queue_age:<τ>:<α>:<P>'   — queue-M with exp age decay: item votes
                              exp(-(now−t_submit)/τ); the stuck-item
                              'cemetery' decays out instead of dominating
- 'queuecap:<D>:<α>:<P>'    — queue-M over placeable classes only: items
                              with gpu_demand_total > D never vote
- 'running:<w>:<α>:<prior>' — α·running-set histogram + (1−α)·prior;
                              w = held (time held so far) | count | dur
- 'mix:<w>@<s>|<w>@<s>|..'  — convex mixture over full sub-specs (static or
                              tracking); re-emits when any term updates

`parse_mspec` is pure string → MSpec; resolution against an actual stream
lives in `workload/resolve.py` (needs items/header/M).
"""

from dataclasses import dataclass
from pathlib import Path

TRACKING_KINDS = ("lead", "trail", "trail_occ", "ewma", "queue", "queueage",
                  "queuecap", "running", "mix")
STATIC_KINDS = ("stream", "empirical", "occupancy", "tilted", "stale",
                "stale_occ")


@dataclass(frozen=True)
class MSpec:
    """Parsed M spec. `prior` is the nested spec for ewma/queue arms."""

    kind: str
    slug: str
    gamma: float | None = None
    source: str | None = None
    window_s: float | None = None
    eta: float | None = None
    alpha: float | None = None
    tau_s: float | None = None
    cap: float | None = None
    prior: "MSpec | None" = None
    terms: "tuple[tuple[float, MSpec], ...] | None" = None

    @property
    def tracking(self) -> bool:
        return self.kind in TRACKING_KINDS


def parse_mspec(spec: str) -> MSpec:
    kind, sep, arg = spec.partition(":")
    if spec in ("stream", "empirical", "occupancy"):
        return MSpec(kind=spec, slug={"occupancy": "occ"}.get(spec, spec))
    if kind == "tilted" and sep:
        g_s, _, prior_spec = arg.partition(":")
        g = float(g_s)
        if prior_spec:
            prior = parse_mspec(prior_spec)
            if prior.tracking:
                raise ValueError(
                    "tilted:<g>:<prior> needs a static prior (tracking M "
                    "changes support over time — tilt it downstream)")
            return MSpec(kind="tilted", slug=f"tilt{g:g}-{prior.slug}",
                         gamma=g, prior=prior)
        return MSpec(kind="tilted", slug=f"tilt{g:g}", gamma=g)
    if kind == "stale" and sep:
        if not arg:
            raise ValueError("stale:<file> requires a path")
        return MSpec(kind="stale", slug=f"stale-{Path(arg).stem}", source=arg)
    if kind == "stale_occ" and sep:
        if not arg:
            raise ValueError("stale_occ:<file> requires a path")
        return MSpec(kind="stale_occ", slug=f"stocc-{Path(arg).stem}", source=arg)
    if kind == "slide_trail_occ" and sep:
        w_s, _, src = arg.partition(":")
        if not src:
            raise ValueError(
                "slide_trail_occ needs a duration source: "
                "slide_trail_occ:<s>:<self|file.jsonl|file.csv>"
            )
        w = float(w_s)
        if w <= 0:
            raise ValueError(f"sliding window must be > 0 s, got {w}")
        stag = "self" if src == "self" else Path(src).stem
        return MSpec(kind="trail_occ", slug=f"socc{w:g}-{stag}",
                     window_s=w, source=src)
    if kind in ("slide_lead", "slide_trail") and sep:
        w = float(arg)
        if w <= 0:
            raise ValueError(f"sliding window must be > 0 s, got {w}")
        mode = kind.split("_")[1]
        return MSpec(kind=mode, slug=f"s{mode}{w:g}", window_s=w)
    if kind == "slide_ewma" and sep:
        eta_s, _, prior_spec = arg.partition(":")
        if not prior_spec:
            raise ValueError(
                "slide_ewma needs a seed prior: "
                "slide_ewma:<eta>:<prior-spec> (e.g. stale:<file>)"
            )
        eta = float(eta_s)
        if not 0 < eta <= 1:
            raise ValueError(f"ewma eta must be in (0, 1], got {eta}")
        prior = parse_mspec(prior_spec)
        return MSpec(kind="ewma", slug=f"sewma{eta:g}-{prior.slug}", eta=eta, prior=prior)
    if kind == "queue" and sep:
        a_s, _, prior_spec = arg.partition(":")
        if not prior_spec:
            raise ValueError(
                "queue M needs a blend prior: "
                "queue:<alpha>:<prior-spec> (e.g. stale:<file>)"
            )
        alpha = float(a_s)
        if not 0 < alpha <= 1:
            raise ValueError(f"queue alpha must be in (0, 1], got {alpha}")
        prior = parse_mspec(prior_spec)
        return MSpec(kind="queue", slug=f"que{alpha:g}-{prior.slug}", alpha=alpha, prior=prior)
    if kind == "queue_age" and sep:
        tau_s, _, rest = arg.partition(":")
        a_s, _, prior_spec = rest.partition(":")
        if not prior_spec:
            raise ValueError(
                "queue_age needs tau, alpha, prior: "
                "queue_age:<tau-s>:<alpha>:<prior-spec>")
        tau = float(tau_s)
        if not tau > 0:
            raise ValueError(f"queue_age tau must be > 0 s, got {tau}")
        alpha = float(a_s)
        if not 0 < alpha <= 1:
            raise ValueError(f"queue_age alpha must be in (0, 1], got {alpha}")
        prior = parse_mspec(prior_spec)
        return MSpec(kind="queueage",
                     slug=f"qage{tau:g}_{alpha:g}-{prior.slug}",
                     tau_s=tau, alpha=alpha, prior=prior)
    if kind == "queuecap" and sep:
        cap_s, _, rest = arg.partition(":")
        a_s, _, prior_spec = rest.partition(":")
        if not prior_spec:
            raise ValueError(
                "queuecap needs cap, alpha, prior: "
                "queuecap:<gpu-demand-cap>:<alpha>:<prior-spec>")
        cap = float(cap_s)
        if not cap > 0:
            raise ValueError(f"queuecap cap must be > 0, got {cap}")
        alpha = float(a_s)
        if not 0 < alpha <= 1:
            raise ValueError(f"queuecap alpha must be in (0, 1], got {alpha}")
        prior = parse_mspec(prior_spec)
        return MSpec(kind="queuecap",
                     slug=f"qcap{cap:g}_{alpha:g}-{prior.slug}",
                     cap=cap, alpha=alpha, prior=prior)
    if kind == "running" and sep:
        wmode, _, rest = arg.partition(":")
        if wmode not in ("held", "count", "dur"):
            raise ValueError(
                "running needs a weight mode: "
                "running:<held|count|dur>:<alpha>:<prior-spec>"
            )
        a_s, _, prior_spec = rest.partition(":")
        if not prior_spec:
            raise ValueError(
                "running needs a blend prior: "
                "running:<held|count|dur>:<alpha>:<prior-spec>"
            )
        alpha = float(a_s)
        if not 0 < alpha <= 1:
            raise ValueError(f"running alpha must be in (0, 1], got {alpha}")
        prior = parse_mspec(prior_spec)
        return MSpec(kind="running", slug=f"run{wmode}{alpha:g}-{prior.slug}",
                     alpha=alpha, source=wmode, prior=prior)
    if kind == "mix" and sep:
        terms = []
        for part in arg.split("|"):
            w_s, at, inner = part.partition("@")
            if not at or not inner:
                raise ValueError(
                    "mix terms must be <weight>@<spec> separated by '|' "
                    f"(e.g. mix:0.5@queue:1:empirical|0.5@stream), got {part!r}")
            w = float(w_s)
            if not w > 0:
                raise ValueError(f"mix weight must be > 0, got {w}")
            terms.append((w, parse_mspec(inner)))
        slug = "mix" + "+".join(f"{w:g}_{t.slug}" for w, t in terms)
        return MSpec(kind="mix", slug=slug, terms=tuple(terms))
    raise ValueError(
        f"unknown policy-m spec {spec!r} — want stream|empirical|occupancy|"
        "tilted:<gamma>|stale:<file>|stale_occ:<file>|slide_lead:<s>|"
        "slide_trail:<s>|slide_trail_occ:<s>:<dur>|"
        "slide_ewma:<eta>:<prior>|queue:<alpha>:<prior>|"
        "running:<w>:<alpha>:<prior>|mix:<w>@<spec>|..."
    )
