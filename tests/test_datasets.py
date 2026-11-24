"""Dataset-layer tests (schema v2): schema, determinism, honesty layer,
gang-shape mix (default = real PAI v2020 distribution) and fractional
background. Ported from tests/test_gang_dataset.py to the split modules.

Tolerance bounds measured over 10 seeds before being fixed:
|gang/fractional demand share - target| <= 0.026, arrival-class TV vs the policy
workload <= 0.042. Bounds below add a margin.
"""

import dataclasses
import json

import numpy as np
import pytest
from conftest import GANG_DIR, PODS_CSV, needs_gang_streams, needs_trace

from fgdg2.core.tasks import Workload
from fgdg2.datasets.synth import (
    PAI_GANG_SHAPES,
    policy_workload,
    sample_stream,
    tilt_multi_gpu_share,
    workload_for_stream,
)
from fgdg2.datasets.v2023 import (
    build_m_fractional,
    build_m_whole,
    build_workload,
    coarrival_stats,
    load_pods,
    multi_gpu_demand_share,
    whole_gpu_pods,
)
from fgdg2.streams.io import load_stream, workload_fingerprint, write_stream

pytestmark = [needs_trace, needs_gang_streams]

DEMO_NAMES = [
    "synth_gangs_seed0_f00_mix_load1_frac28.jsonl",
    "synth_gangs_seed0_f25_mix_load1_frac28.jsonl",
    "synth_gangs_seed0_f50_mix_load1_frac28.jsonl",
    "synth_gangs_seed0_f50_mix_load1_frac50.jsonl",
    "synth_gangs_seed0_f50_k16_load1.jsonl",
    "synth_gangs_seed0_f50_k16_load1_g50.jsonl",
]
DEMO_STREAMS = [GANG_DIR / n for n in DEMO_NAMES]
SHARE_BOUND = 0.04
TV_BOUND = 0.06


@pytest.fixture(scope="module")
def trace_pods():
    return load_pods(PODS_CSV)


@pytest.fixture(scope="module")
def workloads(trace_pods):
    m_whole, stats = build_m_whole(trace_pods)
    m_frac, frac_stats = build_m_fractional(trace_pods)
    return m_whole, stats, m_frac, frac_stats


def test_committed_demo_streams_exist():
    found = sorted(p.name for p in GANG_DIR.glob("synth_gangs_seed0_*.jsonl"))
    assert found == sorted(DEMO_NAMES), f"unexpected demo set: {found}"
    assert (GANG_DIR / "trace_coarrival_stats.json").exists()
    assert (GANG_DIR / "README.md").exists()


def test_m_whole_stats_and_fingerprint(trace_pods, workloads):
    workload, stats, _, _ = workloads
    assert stats["n_classes"] == 42
    assert stats["kept_mass"] == pytest.approx(1.0)
    assert stats["whole_gpu_demand_share_of_trace"] == pytest.approx(0.7155, abs=5e-3)
    assert stats["mean_worker_gpu_demand"] == pytest.approx(1.093, abs=0.02)
    assert stats["worker_gpu_demands"] == [1.0, 2.0, 4.0, 8.0]
    assert stats["multi_gpu_demand_share"] == pytest.approx(0.102, abs=0.005)
    assert all(c.whole_gpu for c in workload.classes)
    reference, _ = build_workload(whole_gpu_pods(trace_pods), popularity_mass=1.0)
    assert [c.name for c in reference.classes] == [c.name for c in workload.classes]
    assert reference.popularities == pytest.approx(workload.popularities, abs=1e-12)
    assert workload_fingerprint(workload) == workload_fingerprint(reference)


def test_m_fractional_stats(workloads):
    _, _, m_frac, frac_stats = workloads
    assert frac_stats["n_classes"] == 39
    assert frac_stats["fractional_gpu_demand_share_of_trace"] == pytest.approx(0.2845, abs=2e-3)
    assert 0.0 < frac_stats["mean_worker_gpu_demand"] < 1.0
    assert all(0.0 < c.gpu_demand < 1.0 for c in m_frac.classes)
    assert m_frac.popularities.sum() == pytest.approx(1.0)


def test_tilt_multi_gpu_share_math(workloads):
    workload, _, _, _ = workloads
    tilted = tilt_multi_gpu_share(workload, 0.5)
    assert tilted.popularities.sum() == pytest.approx(1.0)
    assert multi_gpu_demand_share(tilted) == pytest.approx(0.5, abs=1e-9)
    assert workload_fingerprint(tilted) != workload_fingerprint(workload)
    with pytest.raises(ValueError):
        tilt_multi_gpu_share(workload, 0.0)
    single_classes = [c for c in workload.classes if c.gpu_demand == 1]
    single_only = Workload(
        single_classes, np.full(len(single_classes), 1.0 / len(single_classes))
    )
    with pytest.raises(ValueError, match="no multi-GPU classes"):
        tilt_multi_gpu_share(single_only, 0.5)


def test_streams_carry_matching_fingerprints(workloads):
    m_whole, _, m_frac, _ = workloads
    for path in DEMO_STREAMS:
        header, _ = load_stream(path, m_whole, m_frac)
        assert header["schema_version"] == 2, path.name
        assert header["m_whole_sha256"] == workload_fingerprint(
            workload_for_stream(m_whole, header)
        ), path.name
        if header["m_fractional_sha256"] is not None:
            assert header["m_fractional_sha256"] == workload_fingerprint(m_frac)
        assert header["note"].startswith("Synthetic gang experiment layer")
        assert header["config"]["capacity"] == 6212.0


def test_policy_workload_mixes_classes(workloads):
    m_whole, _, m_frac, _ = workloads
    by_name = {p.name: p for p in DEMO_STREAMS}
    header, _ = load_stream(by_name[DEMO_NAMES[2]], m_whole, m_frac)  # f50_mix
    wl = policy_workload(m_whole, header, m_frac)
    assert len(wl.classes) == len(m_whole.classes) + len(m_frac.classes)
    assert wl.popularities.sum() == pytest.approx(1.0)
    frac_mass = float(wl.popularities[wl.demand_array < 1.0].sum())
    assert frac_mass > 0.05
    # whole-only legacy stream: no fractional classes mixed in
    header_k16, _ = load_stream(by_name[DEMO_NAMES[4]], m_whole, m_frac)
    wl_k16 = policy_workload(m_whole, header_k16, m_frac)
    assert len(wl_k16.classes) == len(m_whole.classes)


def test_stream_schema_and_classes(workloads):
    m_whole, _, m_frac, _ = workloads
    for path in DEMO_STREAMS:
        header, items = load_stream(path, m_whole, m_frac)
        cfg = header["config"]
        k = cfg.get("gang_size")
        shape_ks = {int(s["k"]) for s in cfg.get("gang_shapes") or []}
        assert [it.arrival_index for it in items] == list(range(len(items)))
        assert header["realized"]["n_items"] == len(items)
        for it in items:
            if it.is_gang:
                assert (k is not None and it.gang_size == k) or it.gang_size in shape_ks
            else:
                assert it.gang_size == 1
            if it.worker_gpu_demand < 1.0:
                assert any(
                    c.cpu_milli == it.worker_cpu_milli
                    and c.gpu_demand == it.worker_gpu_demand
                    for c in m_frac.classes
                )
            else:
                assert it.worker_gpu_demand in (1.0, 2.0, 4.0, 8.0)
            assert it.gang_id == f"g{it.arrival_index:06d}"


def test_non_vacuity(workloads):
    m_whole, _, m_frac, _ = workloads
    by_name = {p.name: p for p in DEMO_STREAMS}
    _, mix_items = load_stream(by_name[DEMO_NAMES[2]], m_whole, m_frac)  # f50_mix
    shape_counts = {}
    for it in mix_items:
        if it.is_gang:
            key = (it.gang_size, it.worker_gpu_demand)
            shape_counts[key] = shape_counts.get(key, 0) + 1
    assert len(shape_counts) >= 4, f"only shapes {shape_counts}"
    assert any(d > 1 for _, d in shape_counts), "no multi-GPU-worker gang shape"
    assert any(k >= 8 for k, _ in shape_counts), "no large-k gang shape"
    _, k16_items = load_stream(by_name[DEMO_NAMES[4]], m_whole, m_frac)
    assert sum(1 for it in k16_items if it.is_gang and it.gang_size == 16) >= 1
    _, f00_items = load_stream(by_name[DEMO_NAMES[0]], m_whole, m_frac)
    assert sum(1 for it in f00_items if it.is_gang) == 0
    assert sum(1 for it in f00_items if 0 < it.worker_gpu_demand < 1) >= 100


def _regenerate(path, m_whole, m_frac, tmp_path):
    header, _ = load_stream(path, m_whole, m_frac)
    meta = {
        "git_sha": header["git_sha"],
        "source_pod_csv": header["source_pod_csv"],
        "source_md5": header["source_pod_md5"],
    }
    cfg = header["config"]
    shapes = (
        tuple((s["k"], s["d"], s["weight"]) for s in cfg["gang_shapes"])
        if cfg.get("gang_shapes")
        else None
    )
    whole = workload_for_stream(m_whole, header)
    frac_share = float(cfg.get("fractional_share") or 0.0)
    new_header, items = sample_stream(
        whole, cfg["gang_fraction"], cfg["load"], cfg["capacity"], header["seed"], meta,
        k=cfg.get("gang_size"), shapes=shapes, multi_gpu_share=cfg.get("multi_gpu_share"),
        fractional_workload=(m_frac if frac_share > 0 else None),
        fractional_share=frac_share,
    )
    out = tmp_path / path.name
    write_stream(out, new_header, items)
    return out


def test_determinism_byte_identical(workloads, tmp_path):
    m_whole, _, m_frac, _ = workloads
    for name in (DEMO_NAMES[1], DEMO_NAMES[4]):  # f25_mix (shapes+frac), f50_k16 (legacy)
        path = GANG_DIR / name
        out = _regenerate(path, m_whole, m_frac, tmp_path)
        assert out.read_bytes() == path.read_bytes(), name


def test_realized_shares_within_tolerance(workloads):
    m_whole, _, m_frac, _ = workloads
    for path in DEMO_STREAMS:
        header, items = load_stream(path, m_whole, m_frac)
        cfg = header["config"]
        gang = sum(it.gang_size * it.worker_gpu_demand for it in items if it.is_gang)
        frac = sum(
            it.worker_gpu_demand
            for it in items
            if not it.is_gang and it.worker_gpu_demand < 1.0
        )
        total = sum(it.gang_size * it.worker_gpu_demand for it in items)
        assert abs(gang / total - cfg["gang_fraction"]) <= SHARE_BOUND, path.name
        assert abs(frac / total - cfg["fractional_share"]) <= SHARE_BOUND, path.name


def test_realized_arrival_class_tv_within_bound(workloads):
    m_whole, _, m_frac, _ = workloads
    for path in DEMO_STREAMS:
        header, items = load_stream(path, m_whole, m_frac)
        wl = policy_workload(m_whole, header, m_frac)
        counts = {}
        for it in items:
            counts[it.class_key] = counts.get(it.class_key, 0) + 1
        n = sum(counts.values())
        tv = 0.5 * sum(
            abs(counts.get((int(c.cpu_milli), float(c.gpu_demand)), 0) / n - float(p))
            for c, p in zip(wl.classes, wl.popularities)
        )
        assert tv <= TV_BOUND, f"{path.name}: TV {tv:.4f} > {TV_BOUND}"


def test_coarrival_stats_match_recomputation(trace_pods):
    committed = json.loads((GANG_DIR / "trace_coarrival_stats.json").read_text())
    recomputed = coarrival_stats(whole_gpu_pods(trace_pods))
    assert committed == json.loads(json.dumps(recomputed))  # JSON round-trip: int keys -> str
    assert committed["group_size_histogram"] == {"1": 3864, "2": 55, "3": 4}
    assert committed["share_pods_in_groups_gt1"] == pytest.approx(0.0306, abs=1e-3)
    assert "NOT verified production gangs" in committed["note"]


def test_load_stream_rejects_bad_class(workloads, tmp_path):
    m_whole, _, m_frac, _ = workloads
    header, items = load_stream(DEMO_STREAMS[0], m_whole, m_frac)
    bad = dataclasses.replace(items[0], worker_cpu_milli=123456)
    out = tmp_path / "bad.jsonl"
    write_stream(out, header, [bad, *items[1:]])
    with pytest.raises(ValueError, match="not in M"):
        load_stream(out, m_whole, m_frac)


def test_pai_shapes_well_formed():
    total = sum(w for _, _, w in PAI_GANG_SHAPES)
    assert total == pytest.approx(1.0, abs=1e-3)
    assert all(d in (1.0, 2.0, 4.0, 8.0, None) for _, d, _ in PAI_GANG_SHAPES)
