"""Shared helpers for reading calibration artifacts and the authors' cached results.

Single home for: artifact paths, policy naming maps, the at-capacity metric accessor
(no silent fallback: artifacts without the key are reported, not misread), and the
reference CSV loader used by make_figs / compare_reference. Works on artifact-v3
files (config/metrics/timeline layout is unchanged).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root — self-contained
RESULTS_DIR = ROOT / "results" / "experiments"
DEFAULT_REFERENCE_CSV = ROOT / "data" / "reference" / "analysis_allo_discrete.csv"

PAPER_ORDER = ["fgd_q", "bestfit", "packing", "dotprod", "clustering", "random"]

REF_POLICY_MAP = {
    "fgd_q": "06-FGD",
    "bestfit": "05-BestFit",
    "packing": "04-GpuPacking",
    "clustering": "03-GpuClustering",
    "dotprod": "02-DotProd",
    "random": "01-Random",
}

STYLE = {
    "fgd_q": ("FGD (quantized)", "#1f77b4", 2.4, "-"),
    "fgd": ("FGD (exact argmin, ours)", "#17becf", 1.6, "--"),
    "bestfit": ("BestFit", "#d62728", 2.0, "-"),
    "packing": ("Packing", "#2ca02c", 1.6, "-"),
    "dotprod": ("DotProd", "#9467bd", 1.6, "-"),
    "clustering": ("Clustering", "#8c564b", 1.6, "-"),
    "random": ("Random", "#7f7f7f", 1.6, ":"),
    "firstfit": ("FirstFit (ours, not in paper)", "#ff7f0e", 1.3, "--"),
}


def run_path(policy: str, seed: int, tag_suffix: str = "", results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / f"calib_{policy}{tag_suffix}_seed{seed}.json"


def load_run(policy: str, seed: int, tag_suffix: str = "", results_dir: Path = RESULTS_DIR):
    path = run_path(policy, seed, tag_suffix, results_dir)
    if not path.exists():
        return None
    art = json.loads(path.read_text())
    if "timeline" not in art:
        return None
    return art


def at_capacity_pct(metrics: dict):
    """unallocated % at the 100%-demand crossing; None when the artifact lacks it
    (legacy schema) — callers must handle None instead of falling back silently."""
    return metrics.get("unallocated_pct_at_capacity")


def stop_pct(metrics: dict):
    return metrics.get("unallocated_pct_of_capacity")


def load_reference(
    policy: str,
    path: Path = DEFAULT_REFERENCE_CSV,
    workload: str = "openb_pod_list_default",
    tune: float = 1.3,
):
    """Authors' cached medians (per integer arrival %) for one policy; None if absent."""
    df = pd.read_csv(path)
    df = df[(df["workload"] == workload) & (df["tune"] == tune)]
    sub = df[df["sc_policy"] == REF_POLICY_MAP.get(policy, policy)]
    if sub.empty:
        return None
    rates = sorted((c for c in df.columns if c.isdigit()), key=int)
    med = sub[rates].median(axis=0)
    return [int(r) for r in rates], [100.0 - float(v) for v in med]
