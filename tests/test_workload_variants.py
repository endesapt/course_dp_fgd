"""Tests for the policy-M variants (direction study 2026-09-22 §3):
tilted / occupancy / stale-source workloads and the d_TV helper.
Ported from tests/test_policy_m_variants.py (spot_gpu -> workload.construct).
"""

import json

import numpy as np
import pandas as pd
import pytest

from fgdg2.core.tasks import TaskClass, Workload
from fgdg2.items import StreamItem
from fgdg2.workload.construct import (
    empirical_workload_from_file,
    occupancy_workload,
    tilted_workload,
    workload_dtv,
)


def make_wl():
    classes = (
        TaskClass("a", cpu_milli=1000, gpu_demand=0.5),
        TaskClass("b", cpu_milli=2000, gpu_demand=1.0, whole_gpu=True),
        TaskClass("c", cpu_milli=4000, gpu_demand=4.0, whole_gpu=True),
        TaskClass("d0", cpu_milli=8000, gpu_demand=0.0),
    )
    return Workload(classes, np.array([0.4, 0.3, 0.2, 0.1]))


def item(cpu, d, dur=10.0, t=0.0, gang=False, k=1):
    return StreamItem(
        gang_id="g", gang_size=k, worker_cpu_milli=cpu,
        worker_gpu_demand=d, arrival_index=0, is_gang=gang,
        submit_time_s=t, duration_s=dur,
    )


def test_tilted_identity_at_gamma0():
    wl = make_wl()
    assert tilted_workload(wl, 0.0) is wl


def test_tilted_shifts_mass_to_large_demand():
    wl = make_wl()
    up = tilted_workload(wl, 1.0)
    assert up.popularities[2] > wl.popularities[2]          # d=4 boosted
    assert up.popularities[1] < wl.popularities[1]          # d=1 relatively down
    down = tilted_workload(wl, -1.0)
    assert down.popularities[2] < wl.popularities[2]
    assert down.popularities[0] > wl.popularities[0]        # d=0.5 boosted
    assert np.isclose(up.popularities.sum(), 1.0)
    assert np.isclose(down.popularities.sum(), 1.0)


def test_tilted_zero_demand_class_is_neutral():
    wl = make_wl()
    up = tilted_workload(wl, 1.0)
    # d=0 class keeps factor 1: its share is renormalized but the RATIO to
    # another unit-factor baseline is preserved — check it did not vanish or
    # blow up relative to the untilted d=1 class weight times its factor.
    raw = wl.popularities * np.array([0.5, 1.0, 4.0, 1.0])
    assert np.allclose(up.popularities, raw / raw.sum())


def test_occupancy_weights_by_duration():
    items = [item(1000, 1.0, dur=10.0), item(1000, 1.0, dur=10.0),
             item(2000, 2.0, dur=60.0)]
    wl = occupancy_workload(items)
    key = {(int(c.cpu_milli), float(c.gpu_demand)): float(p)
           for c, p in zip(wl.classes, wl.popularities)}
    assert np.isclose(key[(1000, 1.0)], 20.0 / 80.0)
    assert np.isclose(key[(2000, 2.0)], 60.0 / 80.0)


def test_occupancy_clips_at_window_end():
    items = [item(1000, 1.0, dur=100.0, t=0.0),
             item(2000, 2.0, dur=100.0, t=90.0)]
    wl = occupancy_workload(items, t_end=100.0)
    key = {(int(c.cpu_milli), float(c.gpu_demand)): float(p)
           for c, p in zip(wl.classes, wl.popularities)}
    # clipped: 100 vs 10
    assert np.isclose(key[(1000, 1.0)], 100.0 / 110.0)
    assert np.isclose(key[(2000, 2.0)], 10.0 / 110.0)


def test_occupancy_requires_durations():
    bad = StreamItem(gang_id="g", gang_size=1, worker_cpu_milli=1000,
                     worker_gpu_demand=1.0, arrival_index=0, is_gang=False)
    with pytest.raises(ValueError, match="duration_s"):
        occupancy_workload([bad])


def test_workload_dtv():
    wl = make_wl()
    assert workload_dtv(wl, wl) == 0.0
    other = Workload(
        (TaskClass("x", cpu_milli=999, gpu_demand=1.0, whole_gpu=True),),
        np.array([1.0]),
    )
    assert workload_dtv(wl, other) == pytest.approx(1.0)


def test_empirical_workload_from_file_jsonl(tmp_path):
    items = [item(1000, 1.0), item(1000, 1.0), item(2000, 4.0)]
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps({"type": "header", "schema_version": 2}) + "\n"
        + "\n".join(json.dumps(it.to_dict()) for it in items) + "\n"
    )
    wl = empirical_workload_from_file(p)
    key = {(int(c.cpu_milli), float(c.gpu_demand)): float(p)
           for c, p in zip(wl.classes, wl.popularities)}
    assert np.isclose(key[(1000, 1.0)], 2 / 3)


def test_empirical_workload_from_file_csv(tmp_path):
    pods = pd.DataFrame(
        {
            "name": ["p1", "p2"],
            "creation_time": [0.0, 1.0],
            "cpu_milli": [1000, 2000],
            "num_gpu": [1, 4],
            "gpu_milli": [1000, 1000],
            "memory_mib": [0, 0],
        }
    )
    p = tmp_path / "pods.csv"
    pods.to_csv(p, index=False)
    wl = empirical_workload_from_file(p)
    key = {(int(c.cpu_milli), float(c.gpu_demand)): float(p)
           for c, p in zip(wl.classes, wl.popularities)}
    assert key == {(1000, 1.0): 0.5, (2000, 4.0): 0.5}
