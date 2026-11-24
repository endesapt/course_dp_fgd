"""PAI v2020 adapter tests — synthetic tables plus a real-data integration check."""

import pandas as pd
import pytest
from conftest import PAI_JOB_CSV, PAI_MACHINE_CSV, PAI_TASK_CSV, needs_pai

from fgdg2.datasets.pai import (
    domain_capacity,
    load_job_tasks,
    load_machine_spec,
    max_worker_gpus,
    nodes_to_node_frame,
    split_tasks,
    stream_header,
    task_kind,
    tasks_to_items,
    tasks_to_pod_frame,
)
from fgdg2.datasets.synth import workload_for_stream
from fgdg2.datasets.v2023 import build_m_fractional, build_m_whole
from fgdg2.streams.io import load_stream, verify_fingerprints, write_stream


@pytest.fixture()
def tiny_tables(tmp_path):
    jobs = pd.DataFrame(
        [
            ("j1", "id1", "u1", "Terminated", 100.0, 200.0),
            ("j2", "id2", "u2", "Terminated", 150.0, 250.0),
            ("j3", "id3", "u3", "Running", 300.0, None),
            ("j4", "id4", "u4", "Waiting", None, None),
            ("j5", "id5", "u5", "Terminated", 400.0, 500.0),
        ],
        columns=["job_name", "inst_id", "user", "status", "start_time", "end_time"],
    )
    tasks = pd.DataFrame(
        [
            ("j1", "worker", 2, "Terminated", 110.0, 170.0, 600.0, 30.0, 100.0, "T4"),
            ("j1", "ps", 1, "Terminated", 110.0, 170.0, 100.0, 10.0, 0.0, "T4"),
            ("j2", "worker", 1, "Terminated", 160.0, 220.0, 1200.0, 60.0, 50.0, "T4"),
            ("j3", "worker", 1, "Running", 310.0, None, 600.0, 30.0, 200.0, "T4"),
            ("j4", "worker", 2, "Waiting", None, None, 600.0, 30.0, 100.0, "T4"),
            ("j5", "worker", 2, "Terminated", 410.0, 450.0, 600.0, 30.0, 150.0, "T4"),
            ("j5", "fracgang", 2, "Terminated", 410.0, 450.0, 600.0, 30.0, 50.0, "T4"),
            ("j5", "v100", 1, "Terminated", 410.0, 450.0, 600.0, 30.0, 100.0, "V100"),
            ("j5", "toolarge", 1, "Terminated", 410.0, 450.0, 600.0, 30.0, 300.0, "T4"),
            ("j5", "notype", 1, "Terminated", 410.0, 450.0, 600.0, 30.0, 100.0, None),
        ],
        columns=[
            "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
            "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
        ],
    )
    spec = pd.DataFrame(
        [
            ("m1", "T4", 96, 512, 2),
            ("m2", "T4", 96, 512, 2),
            ("m3", "V100", 96, 512, 8),
            ("m4", "CPU", 96, 512, 0),
        ],
        columns=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
    )
    paths = {}
    for name, frame in (("job", jobs), ("task", tasks), ("machine", spec)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, header=False, index=False)
        paths[name] = path
    return paths


def test_load_and_join(tiny_tables):
    df = load_job_tasks(tiny_tables["job"], tiny_tables["task"])
    assert len(df) == 10
    row = df[(df["job_name"] == "j1") & (df["task_name"] == "worker")].iloc[0]
    assert row["submit_time"] == 100.0
    assert row["worker_num"] == 2
    assert row["cpu_request"] == pytest.approx(6.0)
    assert row["gpu_request"] == pytest.approx(1.0)
    assert row["duration"] == pytest.approx(60.0)
    assert row["wait_s"] == pytest.approx(10.0)


def test_split_tasks_drops(tiny_tables):
    spec = load_machine_spec(tiny_tables["machine"])
    df = load_job_tasks(tiny_tables["job"], tiny_tables["task"])
    t4 = df[df["gpu_type"] == "T4"]
    kept, drops = split_tasks(t4, max_worker_gpus(spec))
    assert sorted(kept["job_name"] + ":" + kept["task_name"]) == [
        "j1:worker", "j2:worker", "j3:worker",
    ]
    assert drops["dropped_zero_gpu"] == 1          # j1:ps
    assert drops["dropped_missing_submit"] == 1    # j4:worker
    assert drops["dropped_unsupported_combined"] == 1  # j5:worker (150%)
    assert drops["dropped_fractional_gangs"] == 1  # j5:fracgang
    assert drops["dropped_too_large_for_domain"] == 1  # j5:toolarge (3 > 2)
    _, all_drops = split_tasks(df, max_worker_gpus(spec))
    assert all_drops["dropped_missing_gpu_type"] == 1  # j5:notype


def test_nodes_and_capacity(tiny_tables):
    spec = load_machine_spec(tiny_tables["machine"])
    assert max_worker_gpus(spec)["T4"] == 2
    assert domain_capacity(spec, "T4") == 4
    nodes = nodes_to_node_frame(spec, "T4")
    assert list(nodes.columns) == ["sn", "gpu", "model", "cpu_milli", "memory_mib"]
    assert nodes["sn"].tolist() == ["m1", "m2"]
    assert nodes["gpu"].tolist() == [2, 2]


def test_items_and_cap(tiny_tables):
    spec = load_machine_spec(tiny_tables["machine"])
    df = load_job_tasks(tiny_tables["job"], tiny_tables["task"])
    kept, _ = split_tasks(df[df["gpu_type"] == "T4"], max_worker_gpus(spec))
    items = tasks_to_items(kept)
    assert [it.gang_id for it in items] == ["j1:worker", "j2:worker", "j3:worker"]
    assert [it.gang_size for it in items] == [2, 1, 1]
    assert items[0].worker_gpu_demand == 1.0
    assert items[1].worker_gpu_demand == 0.5
    assert items[0].duration_s == 60.0
    assert items[2].duration_s is None

    capped = tasks_to_items(kept, capacity=2, load=1.0)
    assert len(capped) == 1
    assert capped[0].gang_id == "j1:worker"


def test_pod_frame_expands_instances(tiny_tables):
    spec = load_machine_spec(tiny_tables["machine"])
    df = load_job_tasks(tiny_tables["job"], tiny_tables["task"])
    kept, _ = split_tasks(df[df["gpu_type"] == "T4"], max_worker_gpus(spec))
    pods = tasks_to_pod_frame(kept)
    assert len(pods) == 4  # j1 has 2 instances, j2/j3 one each
    j1 = pods[pods["name"].str.startswith("j1:worker:")]
    assert len(j1) == 2
    assert set(j1["num_gpu"]) == {1}
    assert set(j1["gpu_milli"]) == {1000}
    j2 = pods[pods["name"].str.startswith("j2:worker:")]
    assert len(j2) == 1
    assert j2["num_gpu"].iloc[0] == 1
    assert j2["gpu_milli"].iloc[0] == 500

    m_whole, _ = build_m_whole(pods)
    m_frac, _ = build_m_fractional(pods)
    assert {c.gpu_demand for c in m_whole.classes} == {1.0, 2.0}
    assert {c.gpu_demand for c in m_frac.classes} == {0.5}


def test_stream_roundtrip(tiny_tables, tmp_path):
    spec = load_machine_spec(tiny_tables["machine"])
    df = load_job_tasks(tiny_tables["job"], tiny_tables["task"])
    kept, drops = split_tasks(df[df["gpu_type"] == "T4"], max_worker_gpus(spec))
    items = tasks_to_items(kept)
    pods = tasks_to_pod_frame(kept)
    m_whole, _ = build_m_whole(pods)
    m_frac, _ = build_m_fractional(pods)
    header = stream_header(
        gpu_type="T4", capacity=4.0, load=None, items=items,
        m_whole=m_whole, m_frac=m_frac,
        meta={"git_sha": "test", "source_task_csv": "task.csv", "source_task_md5": "x"},
        filter_stats=drops,
    )
    path = tmp_path / "stream.jsonl"
    write_stream(path, header, items)
    got_header, got_items = load_stream(path, m_whole, m_frac)
    verify_fingerprints(path, got_header, workload_for_stream(m_whole, got_header), m_frac)
    assert got_header["config"]["real_trace"] == "pai-v2020"
    assert [it.gang_id for it in got_items] == ["j1:worker", "j2:worker", "j3:worker"]


@needs_pai
def test_real_pai_population():
    spec = load_machine_spec(PAI_MACHINE_CSV)
    assert len(spec) == 1897
    assert int(spec["cap_gpu"].sum()) == 6742
    df = load_job_tasks(PAI_JOB_CSV, PAI_TASK_CSV)
    assert len(df) == 1_261_050
    whole = df[df["plan_gpu"].map(task_kind) == "whole"]
    assert int((whole["worker_num"] > 1).sum()) == 74_068
