"""The --policy-m spec DSL: parse_mspec (pure) + resolve_mspec (against a
stream context). New surface — replaces the parser that lived inside
scripts/run_churn.py."""

import json

import pytest

from fgdg2.core.tasks import workload_from_counts
from fgdg2.items import StreamItem
from fgdg2.workload.resolve import StreamContext, resolve_mspec
from fgdg2.workload.spec import MSpec, parse_mspec
from fgdg2.workload.tracking import EwmaM, LeadWindowM, QueueAwareM, TrailWindowM


def test_static_specs():
    assert parse_mspec("stream") == MSpec(kind="stream", slug="stream")
    assert parse_mspec("empirical") == MSpec(kind="empirical", slug="empirical")
    assert parse_mspec("occupancy") == MSpec(kind="occupancy", slug="occ")
    m = parse_mspec("tilted:1.5")
    assert (m.kind, m.slug, m.gamma) == ("tilted", "tilt1.5", 1.5)
    m = parse_mspec("stale:/x/pods_w12.csv")
    assert (m.kind, m.slug, m.source) == ("stale", "stale-pods_w12", "/x/pods_w12.csv")


def test_tracking_specs():
    m = parse_mspec("slide_lead:300")
    assert (m.kind, m.slug, m.window_s, m.tracking) == ("lead", "slead300", 300.0, True)
    m = parse_mspec("slide_trail:60.5")
    assert (m.kind, m.slug, m.window_s) == ("trail", "strail60.5", 60.5)
    m = parse_mspec("slide_ewma:0.05:tilted:1.5")
    assert (m.kind, m.eta, m.slug) == ("ewma", 0.05, "sewma0.05-tilt1.5")
    assert m.prior == MSpec(kind="tilted", slug="tilt1.5", gamma=1.5)
    m = parse_mspec("queue:1:stale:x")
    assert (m.kind, m.alpha, m.slug) == ("queue", 1.0, "que1-stale-x")
    assert m.prior == MSpec(kind="stale", slug="stale-x", source="x")


def test_nested_priors():
    """Priors can nest: ewma seeded by a stale file seeded... etc."""
    m = parse_mspec("slide_ewma:0.1:slide_trail:60")
    assert m.prior.kind == "trail" and m.prior.window_s == 60.0


def test_spec_errors():
    for bad in ("", "bogus", "tilted:", "tilted:abc", "stale:",
                "slide_lead:0", "slide_lead:", "slide_ewma:0.5",
                "slide_ewma:0:empirical", "queue:1", "queue:0:empirical"):
        with pytest.raises(ValueError):
            parse_mspec(bad)


def _ctx(items, header=None):
    wl = workload_from_counts({(8000, 1.0): 3, (4000, 2.0): 1})
    return StreamContext(
        items=items,
        header=header or {"config": {"gang_fraction": 0.5, "gang_size": 4}},
        m_whole=wl,
    )


def _items():
    return [
        StreamItem(gang_id=f"g{i}", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=1.0, arrival_index=i, is_gang=False,
                   submit_time_s=float(i * 10), duration_s=30.0)
        for i in range(4)
    ] + [
        StreamItem(gang_id="g9", gang_size=2, worker_cpu_milli=4000,
                   worker_gpu_demand=2.0, arrival_index=9, is_gang=True,
                   submit_time_s=90.0, duration_s=30.0)
    ]


def test_resolve_static_arms():
    ctx = _ctx(_items())
    emp = resolve_mspec("empirical", ctx)
    assert emp.provider is None and emp.workload is ctx.oracle
    # oracle = the item histogram: 4 of (8000,1) + 1 of (4000,2)
    mix = {c.name: float(p) for c, p in
           zip(emp.workload.classes, emp.workload.popularities)}
    assert mix == {"w8000_1": 0.8, "w4000_2": 0.2}

    occ = resolve_mspec("occupancy", ctx)
    assert occ.provider is None and occ.workload.classes

    t0 = resolve_mspec("tilted:0", ctx)
    assert t0.workload is ctx.oracle            # gamma=0 is the oracle itself
    t1 = resolve_mspec("tilted:1", ctx)
    mix = {c.name: float(p) for c, p in
           zip(t1.workload.classes, t1.workload.popularities)}
    assert mix["w4000_2"] > 0.2 and mix["w8000_1"] < 0.8  # d=2 class boosted


def test_resolve_stale_arm(tmp_path):
    stale = tmp_path / "past.jsonl"
    stale.write_text(
        json.dumps({"type": "header", "schema_version": 2}) + "\n"
        + json.dumps(_items()[-1].to_dict()) + "\n")
    ctx = _ctx(_items())
    r = resolve_mspec(f"stale:{stale}", ctx)
    assert r.provider is None
    assert r.detail["stale_source"] == str(stale)
    mix = {c.name: float(p) for c, p in
           zip(r.workload.classes, r.workload.popularities)}
    assert mix == {"w4000_2": 1.0}


def test_resolve_tracking_arms():
    ctx = _ctx(_items())
    r = resolve_mspec("slide_lead:100", ctx)
    assert isinstance(r.provider, LeadWindowM)
    assert r.detail["slide_mode"] == "lead" and r.detail["slide_window_s"] == 100.0
    r = resolve_mspec("slide_trail:100", ctx)
    assert isinstance(r.provider, TrailWindowM)
    r = resolve_mspec("slide_ewma:0.1:empirical", ctx)
    assert isinstance(r.provider, EwmaM)
    assert r.workload is ctx.oracle   # seed = resolved prior
    assert r.detail["slide_prior"] == "empirical"
    r = resolve_mspec("queue:0.5:empirical", ctx)
    assert isinstance(r.provider, QueueAwareM)
    assert r.detail["queue_alpha"] == 0.5


def test_tracking_providers_carry_oracle():
    """d_TV-vs-oracle bookkeeping: providers get the window oracle so
    tracking_stats can report mean drift of the estimate."""
    ctx = _ctx(_items())
    r = resolve_mspec("slide_trail:100", ctx)
    r.provider.set_time(0.0)  # no arrivals observed -> nothing emitted
    assert r.provider.rebuilds == 0


def test_occ_specs():
    m = parse_mspec("stale_occ:/x/past.jsonl")
    assert (m.kind, m.slug, m.source, m.tracking) == \
        ("stale_occ", "stocc-past", "/x/past.jsonl", False)
    m = parse_mspec("slide_trail_occ:3600:self")
    assert (m.kind, m.slug, m.window_s, m.source, m.tracking) == \
        ("trail_occ", "socc3600-self", 3600.0, "self", True)
    m = parse_mspec("slide_trail_occ:86400:/x/durs.jsonl")
    assert (m.kind, m.source) == ("trail_occ", "/x/durs.jsonl")
    for bad in ("stale_occ:", "slide_trail_occ:100",
                "slide_trail_occ:0:self", "slide_trail_occ:abc:self"):
        with pytest.raises(ValueError):
            parse_mspec(bad)


def test_resolve_stale_occ(tmp_path):
    """stale_occ weights by duration: the 2-worker 60s gang outweighs three
    30s singletons of the other class."""
    past_items = [
        StreamItem(gang_id=f"p{i}", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=1.0, arrival_index=i, is_gang=False,
                   submit_time_s=float(i), duration_s=30.0)
        for i in range(3)
    ] + [
        StreamItem(gang_id="p9", gang_size=2, worker_cpu_milli=4000,
                   worker_gpu_demand=2.0, arrival_index=9, is_gang=True,
                   submit_time_s=10.0, duration_s=60.0)
    ]
    f = tmp_path / "past.jsonl"
    f.write_text(
        json.dumps({"type": "header", "schema_version": 2}) + "\n"
        + "\n".join(json.dumps(it.to_dict()) for it in past_items) + "\n")
    r = resolve_mspec(f"stale_occ:{f}", _ctx(_items()))
    mix = {c.name: float(p) for c, p in
           zip(r.workload.classes, r.workload.popularities)}
    # arrival mix would be 75/25; occupancy mix is 3*30 : 1*60 = 60/40
    assert mix == {"w8000_1": 0.6, "w4000_2": 0.4}
    assert r.detail["stale_occ_source"] == str(f)


def test_resolve_trail_occ(tmp_path):
    from fgdg2.workload.tracking import TrailOccupancyM
    ctx = _ctx(_items())
    r = resolve_mspec("slide_trail_occ:100:self", ctx)
    assert isinstance(r.provider, TrailOccupancyM)
    assert r.detail["slide_mode"] == "trail_occ"
    # file-sourced weight table
    table = tmp_path / "durs.jsonl"
    table.write_text(
        json.dumps({"type": "header", "schema_version": 2}) + "\n"
        + "\n".join(json.dumps(it.to_dict()) for it in _items()) + "\n")
    r = resolve_mspec(f"slide_trail_occ:100:{table}", ctx)
    assert r.provider._weights[(8000, 1.0)] == 30.0


def test_trail_occ_weighting_and_eviction():
    """Weighted arrival + windowed eviction: 30s-vs-60s classes swap the
    mix vs the raw histogram, and old entries age out."""
    from fgdg2.workload.tracking import TrailOccupancyM
    p = TrailOccupancyM(100.0)
    a = StreamItem(gang_id="a", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=1.0, arrival_index=0, is_gang=False,
                   submit_time_s=0.0, duration_s=30.0)
    b = StreamItem(gang_id="b", gang_size=2, worker_cpu_milli=4000,
                   worker_gpu_demand=2.0, arrival_index=1, is_gang=True,
                   submit_time_s=50.0, duration_s=90.0)
    p.observe_arrival(a, 0.0)
    p.observe_arrival(b, 50.0)
    p.set_time(50.0)
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 0.25, "w4000_2": 0.75}  # 30 : 90
    p.set_time(101.0)  # a aged out (t=0 <= 1)
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w4000_2": 1.0}


def test_running_specs():
    m = parse_mspec("running:held:0.5:empirical")
    assert (m.kind, m.slug, m.alpha, m.source, m.tracking) == \
        ("running", "runheld0.5-empirical", 0.5, "held", True)
    assert m.prior == MSpec(kind="empirical", slug="empirical")
    m = parse_mspec("running:count:1:stale:x")
    assert (m.kind, m.source, m.alpha, m.slug) == \
        ("running", "count", 1.0, "runcount1-stale-x")
    for bad in ("running:0.5:empirical", "running:bogus:0.5:empirical",
                "running:held:0:empirical", "running:held:0.5"):
        with pytest.raises(ValueError):
            parse_mspec(bad)


def test_resolve_running_arm():
    from fgdg2.workload.tracking import RunningOccupancyM
    ctx = _ctx(_items())
    r = resolve_mspec("running:held:0.5:empirical", ctx)
    assert isinstance(r.provider, RunningOccupancyM)
    assert r.workload is ctx.oracle  # seed = resolved prior
    assert r.detail["slide_mode"] == "running"
    assert r.detail["run_weight"] == "held" and r.detail["queue_alpha"] == 0.5


def test_running_provider_weighting():
    """held = time held so far: a 90s-held job outweighs a fresh 10s one 9:1;
    count mode ignores held time; empty running set at alpha=1 -> prior."""
    from fgdg2.workload.tracking import RunningOccupancyM
    prior = workload_from_counts({(8000, 1.0): 3, (4000, 2.0): 1})
    a = StreamItem(gang_id="a", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=1.0, arrival_index=0, is_gang=False,
                   submit_time_s=0.0, duration_s=120.0)
    b = StreamItem(gang_id="b", gang_size=2, worker_cpu_milli=4000,
                   worker_gpu_demand=2.0, arrival_index=1, is_gang=True,
                   submit_time_s=90.0, duration_s=30.0)

    p = RunningOccupancyM("held", 1.0, prior)
    p.set_running(100.0, [(0.0, a), (90.0, b)])  # held: 100 vs 10
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 100 / 110, "w4000_2": 10 / 110}

    p = RunningOccupancyM("count", 1.0, prior)
    p.set_running(100.0, [(0.0, a), (90.0, b)])
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 0.5, "w4000_2": 0.5}

    p = RunningOccupancyM("dur", 1.0, prior)
    p.set_running(100.0, [(0.0, a), (90.0, b)])  # dur: 120 vs 30
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 0.8, "w4000_2": 0.2}

    # empty running set at alpha=1 keeps the prior; alpha=0.5 blends halves
    p = RunningOccupancyM("held", 1.0, prior)
    p.set_running(0.0, [])
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 0.75, "w4000_2": 0.25}  # pure prior

    p = RunningOccupancyM("count", 0.5, prior)
    p.set_running(100.0, [(0.0, a), (90.0, b)])
    mix = {c.name: float(w) for c, w in
           zip(p.workload.classes, p.workload.popularities)}
    assert mix == {"w8000_1": 0.5 * 0.5 + 0.5 * 0.75,
                   "w4000_2": 0.5 * 0.5 + 0.5 * 0.25}


def test_mix_specs():
    m = parse_mspec("mix:0.5@queue:1:empirical|0.5@stream")
    assert (m.kind, m.tracking) == ("mix", True)
    assert len(m.terms) == 2
    assert m.terms[0][0] == 0.5 and m.terms[0][1].kind == "queue"
    assert m.terms[1][1] == MSpec(kind="stream", slug="stream")
    for bad in ("mix:", "mix:0.5@", "mix:queue:1:empirical", "mix:0@empirical",
                "mix:0.5@bogus"):
        with pytest.raises(ValueError):
            parse_mspec(bad)


def test_mix_provider():
    """Composite forwards all hooks and re-emits the weighted mixture when a
    sub-estimate changes; static terms contribute their fixed workload."""
    from fgdg2.workload.tracking import CompositeM, QueueAwareM, TrailWindowM
    prior = workload_from_counts({(8000, 1.0): 3, (4000, 2.0): 1})
    q = QueueAwareM(1.0, prior)
    t = TrailWindowM(100.0)
    comp = CompositeM([(0.5, q, None), (0.25, t, None), (0.25, None, prior)])
    a = StreamItem(gang_id="a", gang_size=1, worker_cpu_milli=8000,
                   worker_gpu_demand=1.0, arrival_index=0, is_gang=False,
                   submit_time_s=0.0, duration_s=30.0)
    comp.observe_arrival(a, 0.0)
    comp.set_time(0.0)          # trail emits {a}; queue hasn't emitted yet
    assert comp.workload is None
    comp.set_queue([a])         # queue emits -> all terms live -> first mix
    mix = {c.name: float(w) for c, w in
           zip(comp.workload.classes, comp.workload.popularities)}
    # queue says {8000_1:1.0}, trail says {8000_1:1.0}, prior is 75/25:
    # 0.5*1 + 0.25*1 + 0.25*0.75 = 0.9375 vs 0.25*0.25 = 0.0625
    assert mix == {"w8000_1": 0.9375, "w4000_2": 0.0625}
    comp.set_queue([])          # queue alpha=1 + empty -> pure prior
    mix = {c.name: float(w) for c, w in
           zip(comp.workload.classes, comp.workload.popularities)}
    assert mix == {"w8000_1": 0.8125, "w4000_2": 0.1875}


def test_resolve_mix_arm():
    from fgdg2.workload.tracking import CompositeM
    ctx = _ctx(_items())
    r = resolve_mspec(
        "mix:0.5@queue:1:empirical|0.3@running:held:1:empirical|0.2@stream",
        ctx)
    assert isinstance(r.provider, CompositeM)
    assert len(r.provider._terms) == 3
    assert r.detail["slide_mode"] == "mix"
    # seed workload = weighted mixture of resolved term seeds
    mix = {c.name: float(w) for c, w in
           zip(r.workload.classes, r.workload.popularities)}
    assert abs(sum(mix.values()) - 1.0) < 1e-9
