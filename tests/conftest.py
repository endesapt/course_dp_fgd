"""Shared paths — the repository is self-contained: datasets live in
data/, resolved from here regardless of pytest's cwd."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PODS_CSV = DATA / "trace-gpu-v2023" / "csv" / "openb_pod_list_default.csv"
NODES_CSV = DATA / "trace-gpu-v2023" / "csv" / "openb_node_list_gpu_node.csv"
GANG_DIR = DATA / "gang"
PAI_TASK_CSV = DATA / "trace-gpu-v2020" / "raw" / "pai_task_table.csv"
PAI_JOB_CSV = DATA / "trace-gpu-v2020" / "raw" / "pai_job_table.csv"
PAI_MACHINE_CSV = DATA / "trace-gpu-v2020" / "raw" / "pai_machine_spec.csv"

needs_trace = pytest.mark.skipif(
    not PODS_CSV.exists(), reason="trace-gpu-v2023 CSVs not present"
)
needs_gang_streams = pytest.mark.skipif(
    not GANG_DIR.is_dir(), reason="committed gang demo streams not present"
)
needs_pai = pytest.mark.skipif(
    not (PAI_TASK_CSV.exists() and PAI_JOB_CSV.exists() and PAI_MACHINE_CSV.exists()),
    reason="PAI v2020 raw tables not present",
)
