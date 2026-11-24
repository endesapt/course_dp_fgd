"""Experiment runners — the script logic as testable functions.

Each takes a frozen config (experiments/config.py), writes artifact-v3 records
+ a summary CSV, and returns the summary rows. Scripts only parse argv.
"""

import csv
import glob
import os
import time
from pathlib import Path

import numpy as np

from ..core.cluster import Cluster
from ..core.node import Node
from ..datasets.durations import assign_durations, duration_samples
from ..datasets.synth import workload_for_stream
from ..datasets.v2023 import (
    POPULARITY_MASS_THRESHOLD,
    build_m_fractional,
    build_m_whole,
    build_workload,
    load_nodes,
    load_pods,
    multi_gpu_demand_share,
)
from ..io.artifacts import make_artifact, write_artifact
from ..metrics.results import streak_count
from ..metrics.timeline import (
    downsample_demand_timeline,
    downsample_event_timeline,
)
from ..policies.dynamic import DynamicMPolicy
from ..policies.registry import (
    GANG_SEMANTICS,
    build_inflation_policy,
    build_mixed_policy,
)
from ..sim.churn import QUEUE_MODES, ChurnRunner
from ..sim.elastic import ElasticConfig, ElasticPool
from ..sim.gang import GangRunner
from ..sim.inflation import InflationRunner
from ..streams.io import load_stream, verify_fingerprints
from ..workload.construct import workload_dtv
from ..workload.resolve import StreamContext, resolve_mspec
from ..workload.spec import parse_mspec
from .config import ChurnConfig, GangConfig, InflationConfig


def _summary_csv(path: Path, fields: list[str], rows: dict) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in sorted(rows, key=str):
            w.writerow({f: rows[key].get(f, "") for f in fields})
    os.replace(tmp, path)


def _replicated_cluster(nodes_csv: str, replicas: int) -> Cluster:
    cluster = load_nodes(nodes_csv)
    if replicas <= 1:
        return cluster
    return Cluster([
        Node(f"{n.id}#r{r}", n.num_gpus, model=n.model,
             cpu_milli=n.cpu_milli, memory_mib=n.memory_mib)
        for n in cluster.nodes for r in range(replicas)
    ])


def _load_stream_checked(stream_path: Path, m_whole, m_frac):
    """load_stream + fingerprint gate (pods↔stream pairing, audit A1)."""
    header, items = load_stream(stream_path, m_whole, m_frac)
    verify_fingerprints(stream_path, header, workload_for_stream(m_whole, header), m_frac)
    return header, items


# -- Layer-1: inflation calibration -------------------------------------------

def run_inflation(cfg: InflationConfig, log=print) -> list[dict]:
    """Paper §6.1 inflation benchmark — one artifact per (policy, seed)."""
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pods = load_pods(cfg.pods_csv)
    workload, pods = build_workload(pods)
    capacity = load_nodes(cfg.nodes_csv).total_gpus
    log(f"nodes={len(load_nodes(cfg.nodes_csv).nodes)} capacity={capacity} "
        f"classes={len(workload.classes)}")

    tag = cfg.policy if cfg.sampling == "replacement" else f"{cfg.policy}_{cfg.sampling}"
    if cfg.stop_fraction != 1.0:
        tag = f"{tag}_sf{cfg.stop_fraction:g}"
    rows = []
    for seed in range(cfg.seeds):
        cluster = load_nodes(cfg.nodes_csv)
        policy = build_inflation_policy(cfg.policy, workload, cluster, seed, fast=cfg.fast)
        runner = InflationRunner(cluster, policy, capacity=capacity,
                                 stop_fraction=cfg.stop_fraction)
        t0 = time.time()
        res = runner.run(pods, seed=seed, sampling=cfg.sampling)
        wall = time.time() - t0
        config = {
            "policy": cfg.policy, "seed": seed, "capacity": capacity,
            "mode": f"{cfg.sampling}_{'fast' if cfg.fast else 'naive'}",
            "sampling": cfg.sampling, "stop_fraction": cfg.stop_fraction,
            "quantized": cfg.policy == "fgd_q",
            "n_nodes": len(cluster.nodes), "n_classes": len(workload.classes),
            "popularity_mass_threshold": POPULARITY_MASS_THRESHOLD,
            "pods_csv": cfg.pods_csv, "nodes_csv": cfg.nodes_csv,
        }
        metrics = {
            "placed": res.placed, "unplaced": res.unplaced, "crossed": res.crossed,
            "cumulative_demand": res.cumulative_demand,
            "unallocated_shares_at_crossing": res.unallocated_at_crossing,
            "unallocated_pct_of_capacity": (
                None if res.unallocated_at_crossing is None
                else res.unallocated_at_crossing / capacity * 100),
            "fully_free_at_crossing": res.fully_free_at_crossing,
            "unallocated_shares_at_capacity": res.unallocated_at_capacity,
            "unallocated_pct_at_capacity": (
                None if res.unallocated_at_capacity is None
                else res.unallocated_at_capacity / capacity * 100),
            "fully_free_at_capacity": res.fully_free_at_capacity,
            "decide_runtime": res.decide_runtime_summary,
            "wall_time_s": wall,
        }
        art = make_artifact("inflation", config, metrics,
                            timeline=downsample_demand_timeline(res.timeline, 1200))
        write_artifact(outdir / f"calib_{tag}_seed{seed}.json", art)
        pct_stop = metrics["unallocated_pct_of_capacity"]
        pct_cap = metrics["unallocated_pct_at_capacity"]
        log(f"[{tag} seed{seed}] placed={res.placed} unplaced={res.unplaced} "
            f"unalloc@100%={'n/a' if pct_cap is None else f'{pct_cap:.2f}%'} "
            f"unalloc@stop={'n/a' if pct_stop is None else f'{pct_stop:.1f}%'} "
            f"crossed={res.crossed} wall={wall:.0f}s")
        if not res.crossed:
            log(f"WARNING: [{tag} seed{seed}] did not cross the stop level "
                f"({cfg.stop_fraction:.2f} x capacity); metrics not comparable")
        rows.append({"policy": cfg.policy, "tag": tag, "seed": seed,
                     **{k: metrics[k] for k in ("placed", "unplaced", "crossed")}})
    return rows


# -- Layer-1: gang streams ------------------------------------------------------

def run_gang(cfg: GangConfig, log=print) -> dict:
    """One gang policy over synthetic/real stream files; summary.csv + artifacts."""
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stream_paths = [Path(p) for p in sorted(glob.glob(cfg.streams))]
    if not stream_paths:
        log(f"no stream files match {cfg.streams!r}")
        return {}

    pods = load_pods(cfg.pods_csv)
    m_whole, stats = build_m_whole(pods)
    m_frac, frac_stats = build_m_fractional(pods)
    capacity = load_nodes(cfg.nodes_csv).total_gpus
    log(f"|M_whole|={stats['n_classes']} |M_frac|={frac_stats['n_classes']} "
        f"capacity={capacity} streams={len(stream_paths)}")

    summary_path = outdir / "summary.csv"
    fields = [
        "policy", "background", "semantics", "stream", "fraction", "k", "load",
        "multi_gpu_share", "fractional_share", "stream_seed", "run_seed", "gangs",
        "gangs_placed", "rate", "workers_placed", "unalloc_pct_at_100",
        "unalloc_pct_at_end", "decide_mean_ms", "decide_p99_ms",
        "failure_reasons", "wall_time_s",
    ]
    rows = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as fh:
            rows = {
                (r["policy"], r.get("background", ""), r["stream"], r["run_seed"]): r
                for r in csv.DictReader(fh)
            }

    for stream_path in stream_paths:
        header, items = _load_stream_checked(stream_path, m_whole, m_frac)
        stream_wl = workload_for_stream(m_whole, header)
        ctx = StreamContext(items=items, header=header, m_whole=m_whole, m_frac=m_frac)
        mres = resolve_mspec(cfg.policy_m, ctx)
        if mres.provider is not None:
            raise ValueError(
                f"policy_m={cfg.policy_m!r} is a tracking spec — Layer-1 gang "
                "runs have no time dimension (use stream|empirical|tilted|stale)")
        pwl = mres.workload
        cfgd = header["config"]
        for run_seed in range(cfg.seeds):
            cluster = load_nodes(cfg.nodes_csv)
            rng_gang, rng_bg = (
                np.random.default_rng(s) for s in np.random.SeedSequence(run_seed).spawn(2))
            policy = build_mixed_policy(cfg.policy, pwl, cluster, rng_gang, rng_bg,
                                        background=cfg.background,
                                        beta=cfg.consolidation_beta)
            runner = GangRunner(cluster, policy, capacity=capacity)
            t0 = time.time()
            res = runner.run(items)
            wall = time.time() - t0
            rt = res.decide_runtime_summary
            metrics = {
                "gangs_total": res.gangs_total,
                "gangs_placed": res.gangs_placed,
                "gang_placement_rate": res.gang_placement_rate,
                "workers_placed": res.workers_placed,
                "singletons_total": res.singletons_total,
                "singletons_placed": res.singletons_placed,
                "items_placed": res.items_placed,
                "unplaced_items": res.unplaced_items,
                "cumulative_demand": res.cumulative_demand,
                "unallocated_shares_at_capacity": res.unallocated_at_capacity,
                "unallocated_pct_at_capacity": (
                    None if res.unallocated_at_capacity is None
                    else res.unallocated_at_capacity / capacity * 100),
                "unallocated_shares_at_end": res.unallocated_at_end,
                "unallocated_pct_of_capacity": (
                    None if res.unallocated_at_end is None
                    else res.unallocated_at_end / capacity * 100),
                "failure_reasons": res.failure_reasons,
                "per_shape": res.shape_stats,
                "decide_runtime": rt,
                "wall_time_s": wall,
            }
            config = {
                "policy": cfg.policy,
                "background_policy": cfg.background,
                "gang_semantics": GANG_SEMANTICS[cfg.policy],
                "policy_m": cfg.policy_m,
                "stream_file": str(stream_path).replace("\\", "/"),
                "stream_config": cfgd,
                "stream_seed": header["seed"],
                "run_seed": run_seed,
                "capacity": capacity,
                "n_nodes": len(cluster.nodes),
                "n_classes": len(pwl.classes),
                "popularity_mass_threshold": stats["mass_threshold"],
                "multi_gpu_demand_share": multi_gpu_demand_share(stream_wl),
                "m_whole_sha256": header["m_whole_sha256"],
                "m_fractional_sha256": header.get("m_fractional_sha256"),
                "consolidation_beta": cfg.consolidation_beta,
            }
            art = make_artifact("gang", config, metrics,
                                timeline=downsample_demand_timeline(res.timeline))
            write_artifact(
                outdir / f"calib_gang_{stream_path.stem}_{cfg.policy}_seed{run_seed}.json",
                art)

            rate = res.gang_placement_rate
            rate_s = "n/a" if rate is None else f"{rate:.3f}"
            pct100 = metrics["unallocated_pct_at_capacity"]
            pct100_s = "n/a" if pct100 is None else f"{pct100:.2f}%"
            fails = _top_failures(res.failure_reasons)
            rows[(cfg.policy, cfg.background, stream_path.name, str(run_seed))] = {
                "policy": cfg.policy, "background": cfg.background,
                "semantics": GANG_SEMANTICS[cfg.policy],
                "stream": stream_path.name,
                "fraction": cfgd["gang_fraction"], "k": cfgd["gang_size"],
                "load": cfgd["load"],
                "multi_gpu_share": cfgd.get("multi_gpu_share", ""),
                "fractional_share": cfgd.get("fractional_share", ""),
                "stream_seed": header["seed"], "run_seed": run_seed,
                "gangs": res.gangs_total, "gangs_placed": res.gangs_placed,
                "rate": rate_s, "workers_placed": res.workers_placed,
                "unalloc_pct_at_100": pct100_s,
                "unalloc_pct_at_end": (
                    "" if metrics["unallocated_pct_of_capacity"] is None
                    else round(metrics["unallocated_pct_of_capacity"], 3)),
                "decide_mean_ms": round(rt.get("mean", 0.0), 3),
                "decide_p99_ms": round(rt.get("p99", 0.0), 3),
                "failure_reasons": fails, "wall_time_s": round(wall, 1),
            }
            _summary_csv(summary_path, fields, rows)
            log(f"[{stream_path.stem} {cfg.policy}/{cfg.background} seed{run_seed}] "
                f"gangs={res.gangs_total} placed={res.gangs_placed} rate={rate_s} "
                f"unalloc@100={pct100_s} mean={rt.get('mean', 0.0):.1f}ms "
                f"wall={wall:.0f}s fails={fails or '-'}")
    log(f"saved artifacts + {summary_path}")
    return rows


def _top_failures(reasons: dict, limit=3) -> str:
    if not reasons:
        return ""
    ranked = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return ";".join(f"{name}:{count}" for name, count in ranked)


# -- Layer-2: churn -------------------------------------------------------------

def _elastic_knob_tag(cfg: ChurnConfig) -> str:
    """Non-default elastic knobs → run-tag suffix for artifact names and the
    CSV `elastic` key. Defaults elided so existing tags stay stable.
    h=pin_max_hold_s, b=relaunch_cooldown_s, p=pool cap, q=scan limit,
    bind=provision_binding."""
    t = ""
    if cfg.provision_binding:
        t += "bind"
    if cfg.pin_max_hold_s != 21600.0:
        t += f"h{cfg.pin_max_hold_s:g}"
    if cfg.relaunch_cooldown_s != 3600.0:
        t += f"b{cfg.relaunch_cooldown_s:g}"
    if cfg.pool_max_nodes:
        t += f"p{cfg.pool_max_nodes}"
    if cfg.queue_scan_limit != 256:
        t += f"q{cfg.queue_scan_limit}"
    return t


def run_churn(cfg: ChurnConfig, log=print) -> dict:
    """Temporal replay over stream files; churn_summary.csv + artifacts."""
    parse_mspec(cfg.policy_m)  # fail fast on a malformed M spec
    if cfg.queue_mode not in QUEUE_MODES:
        raise ValueError(f"queue_mode must be one of {QUEUE_MODES}")
    if cfg.elastic and GANG_SEMANTICS[cfg.policy] == "packed":
        # min_fresh_nodes / pin assembly assume spread semantics (k workers
        # over fresh nodes); a packed task needs k free GPUs on ONE node —
        # launch sizing and reservation would silently target the wrong
        # substrate. Packed baselines are static-fleet runs.
        raise ValueError(
            f"policy {cfg.policy} is packed; --elastic supports spread "
            "gang semantics only")
    if cfg.elastic and cfg.queue_mode != "scan":
        # all elastic evidence so far is scan-mode; strict/blocking are
        # untested territory, not a broken combination — warn, don't ban
        log(f"WARNING: --elastic with queue_mode={cfg.queue_mode} is "
            "outside the validated envelope (all published arms are scan)")
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stream_paths = [Path(p) for p in sorted(glob.glob(cfg.streams))]
    if not stream_paths:
        log(f"no stream files match {cfg.streams!r}")
        return {}

    pods = load_pods(cfg.pods_csv)
    m_whole, stats = build_m_whole(pods)
    m_frac, frac_stats = build_m_fractional(pods)
    capacity = load_nodes(cfg.nodes_csv).total_gpus * cfg.node_replicas
    log(f"|M_whole|={stats['n_classes']} |M_frac|={frac_stats['n_classes']} "
        f"capacity={capacity} streams={len(stream_paths)} queue={cfg.queue_mode}")

    samples = fallback = None
    if cfg.durations in ("empirical", "resample"):
        samples, fallback = duration_samples(cfg.duration_source_csv)
        log(f"duration source: {cfg.duration_source_csv} classes={len(samples)}")

    summary_path = outdir / "churn_summary.csv"
    fields = [
        "policy", "background", "policy_m", "queue_mode", "dur_scale", "stream",
        "run_seed", "items", "gangs", "gangs_placed", "rate",
        "singletons_placed", "queued", "never_placed", "max_queue",
        "wait_mean_s", "wait_p50_s", "wait_p90_s", "wait_p99_s",
        "wait_gpu_mean_s", "wait_gpu_h", "gpu_util_pct",
        "assembly_starved_gpu_h", "assembly_starved_pct_capacity",
        "assembly_starved_pct_queue", "assembly_starved_events",
        "unalloc_mean", "unalloc_p90", "empty_nodes_mean",
        "releasable_node_h", "nodes_never_used", "nodes_empty_ge_24h",
        "billed_node_h", "billed_gpu_h", "online_mean",
        "launches", "evictions", "evict_regret", "elastic",
        "beta", "node_replicas", "congested_frac", "decide_attempts",
        "decide_mean_ms", "decide_p99_ms", "wall_time_s",
    ]
    rows = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["policy_m"] = r.get("policy_m") or "empirical"
                rows[(r["policy"], r["policy_m"], r["queue_mode"],
                      r.get("dur_scale", "1.0"), r["stream"], r["run_seed"],
                      r.get("beta", "0"), r.get("node_replicas", "1"),
                      r.get("elastic", ""))] = r

    for stream_path in stream_paths:
        header, items = _load_stream_checked(stream_path, m_whole, m_frac)
        ctx = StreamContext(items=items, header=header, m_whole=m_whole, m_frac=m_frac)
        mres = resolve_mspec(cfg.policy_m, ctx)
        pm_detail = dict(mres.detail)
        pm_dtv = workload_dtv(mres.workload, ctx.oracle)
        m_tag = "" if mres.slug in ("stream", "empirical") else f"_m{mres.slug}"
        if cfg.consolidation_beta:
            m_tag += f"_b{cfg.consolidation_beta:g}"
        if cfg.node_replicas > 1:
            m_tag += f"_x{cfg.node_replicas}"
        if cfg.elastic:
            m_tag += f"_e{cfg.evict_after_s:g}l{cfg.launch_latency_s:g}"
            if cfg.reserve_empty:
                m_tag += f"r{cfg.reserve_empty}"
            if cfg.reserve_top_n:
                m_tag += f"n{cfg.reserve_top_n}"
            if not cfg.cold_start:
                m_tag += "_w"
            m_tag += _elastic_knob_tag(cfg)
        cfgd = header["config"]
        for run_seed in range(cfg.seeds):
            rng = np.random.default_rng(run_seed)
            run_items, n_assigned = assign_durations(
                items, cfg.durations, rng, samples, fallback,
                constant_s=cfg.duration_s, scale=cfg.duration_scale)
            if cfg.elastic:
                # blueprint pool: the CSV is the finite supply; nodes mint
                # fresh incarnations (ids never recycled). Cold start = empty
                # initial cluster; warm = fresh copies of every blueprint.
                # The policy is built against the SUPPLY view — cold-start
                # live clusters are empty and from_cluster() needs dims.
                blueprints = _replicated_cluster(cfg.nodes_csv,
                                                 cfg.node_replicas).nodes
                pool = ElasticPool(ElasticConfig(
                    evict_after_s=cfg.evict_after_s,
                    launch_latency_s=cfg.launch_latency_s,
                    max_nodes=cfg.pool_max_nodes,
                    cold_start=cfg.cold_start,
                    reserve_empty=cfg.reserve_empty,
                    reserve_top_n=cfg.reserve_top_n,
                    pin_max_hold_s=cfg.pin_max_hold_s,
                    relaunch_cooldown_s=cfg.relaunch_cooldown_s,
                    provision_binding=cfg.provision_binding), blueprints)
                cluster = Cluster([]) if cfg.cold_start else Cluster([
                    Node(n.id, n.num_gpus, model=n.model,
                         cpu_milli=n.cpu_milli, memory_mib=n.memory_mib)
                    for n in blueprints])
                policy_cluster = Cluster([
                    Node(n.id, n.num_gpus, model=n.model,
                         cpu_milli=n.cpu_milli, memory_mib=n.memory_mib)
                    for n in blueprints])
            else:
                pool = None
                cluster = _replicated_cluster(cfg.nodes_csv, cfg.node_replicas)
                policy_cluster = cluster
            rng_gang, rng_bg = (
                np.random.default_rng(s) for s in np.random.SeedSequence(run_seed).spawn(2))

            def make_inner(wl, cluster=policy_cluster, rg=rng_gang, rb=rng_bg):
                return build_mixed_policy(cfg.policy, wl, cluster, rg, rb,
                                          background=cfg.background,
                                          beta=cfg.consolidation_beta)

            if mres.provider is not None:
                # tracking arm: the provider rebuilds M_t during the run; the
                # static workload is only the seed prior
                policy = DynamicMPolicy(make_inner, mres.provider)
            else:
                policy = make_inner(mres.workload)
            runner = ChurnRunner(cluster, policy, capacity=capacity,
                                 queue_mode=cfg.queue_mode,
                                 queue_scan_limit=cfg.queue_scan_limit,
                                 elastic=pool)
            t0 = time.time()
            res = runner.run(run_items)
            wall = time.time() - t0
            if mres.provider is not None:
                tstats = policy.tracking_stats()
                pm_detail.update(tstats)
                # a single pre-run d_TV is meaningless for a moving M — report
                # the mean over rebuild states instead
                pm_dtv = tstats["slide_mean_dtv_vs_oracle"]

            tl = res.timeline
            unalloc_arr = np.array([r["unallocated_shares"] for r in tl])
            empty_arr = np.array([r["empty_nodes"] for r in tl])
            ws = res.wait_summary()
            rt = res.decide_runtime_summary()
            config = {
                "policy": cfg.policy,
                "background_policy": cfg.background,
                "gang_semantics": GANG_SEMANTICS[cfg.policy],
                "queue_mode": cfg.queue_mode,
                "queue_scan_limit": cfg.queue_scan_limit,
                "durations": cfg.durations,
                "duration_scale": cfg.duration_scale,
                "durations_assigned": n_assigned,
                "policy_m": cfg.policy_m,
                "policy_m_slug": mres.slug,
                "policy_m_dtv_vs_oracle": pm_dtv,
                "policy_m_detail": pm_detail,
                "consolidation_beta": cfg.consolidation_beta,
                "node_replicas": cfg.node_replicas,
                # elasticity knobs — billing convention: node bills from
                # node_ready until evict; launch latency costs wait, not money
                "elastic": cfg.elastic,
                "evict_after_s": cfg.evict_after_s if cfg.elastic else None,
                "launch_latency_s": cfg.launch_latency_s if cfg.elastic else None,
                "pool_max_nodes": (
                    (cfg.pool_max_nodes or len(blueprints))
                    if cfg.elastic else len(cluster.nodes)),
                "reserve_empty": cfg.reserve_empty if cfg.elastic else None,
                "reserve_top_n": cfg.reserve_top_n if cfg.elastic else None,
                "pin_max_hold_s": cfg.pin_max_hold_s if cfg.elastic else None,
                "relaunch_cooldown_s": (
                    cfg.relaunch_cooldown_s if cfg.elastic else None),
                "provision_binding": (
                    cfg.provision_binding if cfg.elastic else None),
                "cold_start": cfg.cold_start if cfg.elastic else None,
                "stream_file": str(stream_path).replace("\\", "/"),
                "stream_config": cfgd,
                "stream_seed": header["seed"],
                "run_seed": run_seed,
                "capacity": capacity,
                "n_nodes": len(cluster.nodes),
                "measurement_window": {
                    "span_s": res.span,
                    "layer2_window": cfgd.get("layer2_window"),
                },
            }
            metrics = {
                "items_total": res.items_total,
                "gangs_total": res.gangs_total,
                "gangs_placed": res.gangs_placed,
                "gang_placement_rate": res.gang_placement_rate,
                "singletons_total": res.singletons_total,
                "singletons_placed": res.singletons_placed,
                "items_placed": res.items_placed,
                "workers_placed": res.workers_placed,
                "queued_total": res.queued_total,
                "queued_never_placed": res.queued_never_placed,
                "max_queue_depth": res.max_queue_depth,
                "releases": res.releases,
                "wait_summary_s": ws,
                # GPU-demand-weighted wait (big jobs count ∝ their GPUs) +
                # mean GPU utilization = busy GPU-hours / (capacity × window)
                "wait_gpu_mean_s": (
                    res.wait_gpu_s / res.wait_gpu_demand
                    if res.wait_gpu_demand else None),
                "wait_gpu_hours": res.wait_gpu_s / 3600.0,
                "queued_waits_at_end_s": res.queued_waits_end[:50],
                # per-item wait records [wait_s, gpu_demand_total, k, is_gang,
                # t_submit] — per-class wait splits / CDFs / warm-state
                # per-window bucketing; not in the CSV on purpose
                "wait_records": [
                    [round(w, 3), d, k, int(g), round(t, 3)]
                    for w, d, k, g, t in res.wait_records
                ],
                # tracking-M trajectory (hourly [t, workload] snapshots) —
                # postmortem evidence of WHAT the estimator believed when
                "m_history": res.m_history,
                # ∫ queued GPU-demand dt (full run) — queue composition
                "queue_gpu_hours": res.queue_gpu_s / 3600.0,
                "queue_gpu_hours_by_demand": {
                    b: v / 3600.0
                    for b, v in sorted(res.queue_gpu_s_by_bucket.items())
                },
                "assembly_starved_gpu_hours": res.assembly_starved_gpu_s / 3600.0,
                "assembly_starved_pct_capacity": (
                    res.assembly_starved_gpu_s / (capacity * res.span) * 100
                    if capacity and res.span else None),
                "assembly_starved_pct_queue": (
                    res.assembly_starved_gpu_s / res.queue_gpu_s * 100
                    if res.queue_gpu_s else None),
                "assembly_starved_events": res.assembly_starved_events,
                "unalloc_shares_mean": float(unalloc_arr.mean()) if tl else None,
                "unalloc_shares_p90": float(np.quantile(unalloc_arr, 0.9)) if tl else None,
                "unalloc_shares_end": float(unalloc_arr[-1]) if tl else None,
                "unalloc_pct_mean": (
                    float(unalloc_arr.mean() / capacity * 100) if tl else None),
                "empty_nodes_mean": float(empty_arr.mean()) if tl else None,
                "empty_nodes_min": int(empty_arr.min()) if tl else None,
                # time-weighted integrals: releasable node-hours over the
                # ARRIVAL WINDOW (the drain tail is reported separately)
                "releasable_node_hours": res.empty_node_hours / 3600.0,
                "releasable_node_hours_total": res.empty_node_hours_total / 3600.0,
                "unalloc_gpu_hours": res.unalloc_gpu_hours / 3600.0,
                "unalloc_gpu_hours_total": res.unalloc_gpu_hours_total / 3600.0,
                "nodes_never_used": res.nodes_never_used,
                "nodes_empty_ge_1h": streak_count(res, 3600),
                "nodes_empty_ge_6h": streak_count(res, 21600),
                "nodes_empty_ge_24h": streak_count(res, 86400),
                "nodes_empty_ge_72h": streak_count(res, 259200),
                "node_max_empty_streak_s": sorted(
                    res.node_max_empty_streak.values(), reverse=True),
                "busy_gpu_hours": res.busy_gpu_hours / 3600.0,
                # billed = ∫ ONLINE capacity dt (arrival window; static runs
                # reduce to capacity×span — identical to the old denominator)
                "billed_node_hours": res.billed_node_hours / 3600.0,
                "billed_gpu_hours": res.billed_gpu_hours / 3600.0,
                "billed_node_hours_total": res.billed_node_hours_total / 3600.0,
                "billed_gpu_hours_total": res.billed_gpu_hours_total / 3600.0,
                "online_nodes_mean": (
                    res.billed_node_hours / res.arrival_span
                    if res.arrival_span else None),
                "launches": res.launches,
                "evictions": res.evictions,
                "eviction_regret": res.eviction_regret,
                # utilization of PAID capacity: busy / billed (== old
                # capacity×span under a static fleet)
                "gpu_util_mean": (
                    res.busy_gpu_hours / res.billed_gpu_hours
                    if res.billed_gpu_hours else None),
                "congested_frac": res.congested_time / res.span if res.span else None,
                "span_s": res.span,
                "decide_attempts": res.decide_attempts,
                "decide_runtime": rt,
                "wall_time_s": wall,
            }
            ds_tag = "" if cfg.duration_scale == 1.0 else f"_ds{cfg.duration_scale:g}"
            fname = (f"churn_{stream_path.stem}_{cfg.policy}_{cfg.queue_mode}"
                     f"{ds_tag}{m_tag}_seed{run_seed}.json")
            write_artifact(outdir / fname,
                           make_artifact("gang_churn", config, metrics,
                                         timeline=downsample_event_timeline(tl)))

            e_tag = (f"e{cfg.evict_after_s:g}l{cfg.launch_latency_s:g}"
                     f"r{cfg.reserve_empty}n{cfg.reserve_top_n}"
                     f"{'c' if cfg.cold_start else 'w'}"
                     + _elastic_knob_tag(cfg)
                     if cfg.elastic else "")
            rate = res.gang_placement_rate
            row = {
                "policy": cfg.policy, "background": cfg.background,
                "policy_m": mres.slug, "queue_mode": cfg.queue_mode,
                "dur_scale": cfg.duration_scale, "stream": stream_path.name,
                "run_seed": run_seed, "items": res.items_total,
                "gangs": res.gangs_total, "gangs_placed": res.gangs_placed,
                "rate": "n/a" if rate is None else f"{rate:.3f}",
                "singletons_placed": res.singletons_placed,
                "queued": res.queued_total,
                "never_placed": res.queued_never_placed,
                "max_queue": res.max_queue_depth,
                "wait_mean_s": round(ws.get("mean", 0.0), 1),
                "wait_p50_s": round(ws.get("p50", 0.0), 1),
                "wait_p90_s": round(ws.get("p90", 0.0), 1),
                "wait_p99_s": round(ws.get("p99", 0.0), 1),
                "wait_gpu_mean_s": round(
                    res.wait_gpu_s / res.wait_gpu_demand, 1
                ) if res.wait_gpu_demand else "",
                "wait_gpu_h": round(res.wait_gpu_s / 3600.0, 1),
                "assembly_starved_gpu_h": round(
                    res.assembly_starved_gpu_s / 3600.0, 3),
                "assembly_starved_pct_capacity": round(
                    res.assembly_starved_gpu_s / (capacity * res.span) * 100, 3
                ) if capacity and res.span else "",
                "assembly_starved_pct_queue": round(
                    res.assembly_starved_gpu_s / res.queue_gpu_s * 100, 3
                ) if res.queue_gpu_s else "",
                "assembly_starved_events": res.assembly_starved_events,
                "gpu_util_pct": round(
                    100 * res.busy_gpu_hours / res.billed_gpu_hours, 2
                ) if res.billed_gpu_hours else "",
                "unalloc_mean": round(float(unalloc_arr.mean()), 1) if tl else "",
                "unalloc_p90": round(float(np.quantile(unalloc_arr, 0.9)), 1) if tl else "",
                "empty_nodes_mean": round(float(empty_arr.mean()), 1) if tl else "",
                "releasable_node_h": round(res.empty_node_hours / 3600.0, 1),
                "nodes_never_used": res.nodes_never_used,
                "nodes_empty_ge_24h": streak_count(res, 86400),
                "billed_node_h": round(res.billed_node_hours / 3600.0, 1),
                "billed_gpu_h": round(res.billed_gpu_hours / 3600.0, 1),
                "online_mean": round(
                    res.billed_node_hours / res.arrival_span, 2
                ) if res.arrival_span else "",
                "launches": res.launches,
                "evictions": res.evictions,
                "evict_regret": res.eviction_regret,
                "elastic": e_tag,
                "beta": f"{cfg.consolidation_beta:g}",
                "node_replicas": cfg.node_replicas,
                "congested_frac": round(res.congested_time / res.span, 3) if res.span else "",
                "decide_attempts": res.decide_attempts,
                "decide_mean_ms": round(rt.get("mean", 0.0), 3),
                "decide_p99_ms": round(rt.get("p99", 0.0), 3),
                "wall_time_s": round(wall, 1),
            }
            rows[(cfg.policy, mres.slug, cfg.queue_mode, str(cfg.duration_scale),
                  stream_path.name, str(run_seed), f"{cfg.consolidation_beta:g}",
                  str(cfg.node_replicas), e_tag)] = row
            log(f"[{stream_path.stem} {cfg.policy}/{cfg.queue_mode} m={mres.slug} "
                f"dtv={pm_dtv if pm_dtv is not None else float('nan'):.3f} seed{run_seed}] "
                f"gangs={res.gangs_total} placed={res.gangs_placed} rate={row['rate']} "
                f"queued={res.queued_total} never={res.queued_never_placed} "
                f"waitP90={row['wait_p90_s']}s billed={row['billed_node_h']}nh "
                f"starved={row['assembly_starved_gpu_h']}gpu_h "
                f"({row['assembly_starved_pct_capacity']}%cap) "
                f"launch={res.launches} evict={res.evictions} wall={wall:.0f}s")

    _summary_csv(summary_path, fields, rows)
    log(f"saved artifacts + {summary_path}")
    return rows
