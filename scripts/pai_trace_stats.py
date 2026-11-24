#!/usr/bin/env python3
"""Whole-trace composition of PAI 2020 (cluster-trace-gpu-v2020) under the
mixed-block filter (make_pai_blocks.py --mixed), for chapter 1 and appendix A.

Needs the raw tables (data/trace-gpu-v2020/raw of the fgdg2 checkout):
  python scripts/pai_trace_stats.py --raw /path/to/trace-gpu-v2020/raw
Writes results/aggregates/mixed/pai_trace_stats.csv and
results/tables/tabA_pai.tex.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# FGDG2_SRC: fgdg2 checkout with fractional-gang support (branch fractional-gangs)
# until the repo copy of src/ is synced (HANDOFF Q7)
sys.path.insert(0, os.environ.get("FGDG2_SRC", str(ROOT / "src")))

import pandas as pd

from fgdg2.datasets.pai import (
    domain_capacity,
    load_job_tasks,
    load_machine_spec,
    max_worker_gpus,
    split_tasks,
)

DOMAINS = ["MISC", "V100", "V100M32", "P100", "T4"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    raw = Path(args.raw)
    tasks = load_job_tasks(raw / "pai_job_table.csv", raw / "pai_task_table.csv")
    spec = load_machine_spec(raw / "pai_machine_spec.csv")
    maxg = max_worker_gpus(spec)

    with open(raw / "pai_job_table.csv", encoding="utf-8") as fh:
        n_jobs = sum(1 for _ in fh)
    rows = [("jobs", n_jobs), ("task_records", len(tasks)),
            ("machines", len(spec)), ("gpus", int(spec["cap_gpu"].sum()))]
    kept_all = []
    drops_total: dict[str, int] = {}
    for dom in DOMAINS:
        sub = tasks[tasks["gpu_type"] == dom]
        kept, drops = split_tasks(sub, maxg, keep_fractional_gangs=True)
        for k, v in drops.items():
            drops_total[k] = drops_total.get(k, 0) + int(v)
        kept = kept.copy()
        kept["domain"] = dom
        kept["cap"] = domain_capacity(spec, dom)
        kept_all.append(kept)
    k = pd.concat(kept_all, ignore_index=True)
    k["demand"] = k["worker_num"] * k["gpu_request"]
    gang = k["worker_num"] > 1
    frac = k["gpu_request"] < 1.0
    rows += [
        ("kept_tasks_5_domains", len(k)),
        ("kept_gangs", int(gang.sum())),
        ("kept_fractional_gangs", int((gang & frac).sum())),
        ("kept_fractional_singletons", int((~gang & frac).sum())),
        ("kept_whole_tasks", int((~frac).sum())),
        ("gang_share_of_gpu_demand_pct", 100 * k.loc[gang, "demand"].sum() / k["demand"].sum()),
        ("fractional_gang_share_of_gang_demand_pct",
         100 * k.loc[gang & frac, "demand"].sum() / k.loc[gang, "demand"].sum()),
    ]
    rows += [(f"dropped_{n}", v) for n, v in sorted(drops_total.items())]
    per_dom = k.groupby("domain").agg(
        tasks=("demand", "size"), gangs=("worker_num", lambda s: int((s > 1).sum())),
        cap=("cap", "first"))
    fg = k[gang & frac].groupby("domain")["demand"].sum()
    gd = k[gang].groupby("domain")["demand"].sum()
    per_dom["frac_gang_of_gang_demand_pct"] = 100 * fg.reindex(per_dom.index).fillna(0) / gd
    for dom, r in per_dom.iterrows():
        rows += [(f"{dom}_tasks", int(r.tasks)), (f"{dom}_gangs", int(r.gangs)),
                 (f"{dom}_gpus", int(r.cap)),
                 (f"{dom}_frac_gang_of_gang_demand_pct", float(r.frac_gang_of_gang_demand_pct))]

    out = ROOT / "results" / "aggregates" / "mixed"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["stat", "value"])
    df.to_csv(out / "pai_trace_stats.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False))

    v = dict(rows)
    fmt = lambda x: f"{int(x):,}".replace(",", "\\,")
    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\caption{Состав набора данных PAI 2020}",
        "\\label{tab:paicounts}", "\\begin{tabular}{lr}", "\\toprule",
        "Показатель & Значение \\\\", "\\midrule",
        f"Работ & {fmt(v['jobs'])} \\\\",
        f"Записей задач & {fmt(v['task_records'])} \\\\",
        f"Машин / GPU & {fmt(v['machines'])} / {fmt(v['gpus'])} \\\\",
        "\\midrule",
        f"Задач оставлено (5 доменов) & {fmt(v['kept_tasks_5_domains'])} \\\\",
        f"\\quad из них на целые GPU & {fmt(v['kept_whole_tasks'])} \\\\",
        f"\\quad дробных одиночных & {fmt(v['kept_fractional_singletons'])} \\\\",
        f"Групповых задач (\\(k \\ge 2\\)) & {fmt(v['kept_gangs'])} \\\\",
        f"\\quad из них дробных & {fmt(v['kept_fractional_gangs'])} \\\\",
        f"Доля групповых задач в спросе на GPU & {v['gang_share_of_gpu_demand_pct']:.1f}\\% \\\\",
        (f"Доля дробных групповых задач в их спросе & "
         f"{v['fractional_gang_share_of_gang_demand_pct']:.1f}\\% \\\\"),
        "\\midrule",
        f"Отброшено: без GPU & {fmt(v.get('dropped_dropped_zero_gpu', 0))} \\\\",
        f"Отброшено: нецелый запрос > 1 GPU & {fmt(v.get('dropped_dropped_unsupported_combined', 0))} \\\\",
        f"Отброшено: без времени подачи & {fmt(v.get('dropped_dropped_missing_submit', 0))} \\\\",
        f"Отброшено: больше карт, чем на узле & {fmt(v.get('dropped_dropped_too_large_for_domain', 0))} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\caption{PAI 2020 по доменам}",
        "\\label{tab:paidomains}", "\\begin{tabular}{lrrrr}", "\\toprule",
        "Домен & Задач & Групповых & GPU & Дробные групповые, \\% их спроса \\\\", "\\midrule",
    ]
    for dom in DOMAINS:
        lines.append(f"{dom} & {fmt(v[dom + '_tasks'])} & {fmt(v[dom + '_gangs'])} & "
                     f"{fmt(v[dom + '_gpus'])} & {v[dom + '_frac_gang_of_gang_demand_pct']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ROOT / "results" / "tables" / "tabA_pai.tex").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
