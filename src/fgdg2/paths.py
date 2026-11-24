"""Repo-rooted paths — the repository is self-contained (data/, results/),
so defaults resolve HERE, never against the caller's cwd."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/fgdg2/paths.py -> repo root
DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "results"

PODS_CSV = DATA / "trace-gpu-v2023" / "csv" / "openb_pod_list_default.csv"
NODES_CSV = DATA / "trace-gpu-v2023" / "csv" / "openb_node_list_gpu_node.csv"
GANG_STREAMS_GLOB = str(DATA / "gang" / "synth_gangs_seed*_load1.jsonl")
SPOT_DIR = DATA / "spot-gpu-v2026"
REFERENCE_CSV = DATA / "reference" / "analysis_allo_discrete.csv"
