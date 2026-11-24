"""Durations-mode tests — pins the empirical-vs-resample semantics.

`empirical` fills only items MISSING duration_s; on spot-gpu streams every
item carries one, so empirical is a documented no-op there. `resample`
redraws every item from the class-conditional pool — that is the mode that
produces real run-seed variance on duration-carrying streams.
"""

import numpy as np

from fgdg2.datasets.durations import assign_durations
from fgdg2.items import StreamItem


def _item(gid, cpu=8000, d=1.0, dur=100.0, k=1):
    return StreamItem(
        gang_id=gid, gang_size=k, worker_cpu_milli=cpu,
        worker_gpu_demand=d, arrival_index=0, is_gang=k > 1,
        duration_s=dur)


def test_empirical_keeps_stamped_durations():
    items = [_item("a"), _item("b")]
    samples = {(8000, 1.0): np.array([5000.0])}
    out, n = assign_durations(
        items, "empirical", np.random.default_rng(0),
        samples, np.array([5000.0]))
    assert n == 0
    assert [it.duration_s for it in out] == [100.0, 100.0]


def test_resample_redraws_stamped_durations():
    items = [_item("a"), _item("b"), _item("c")]
    samples = {(8000, 1.0): np.array([5000.0, 7000.0])}
    out, n = assign_durations(
        items, "resample", np.random.default_rng(0),
        samples, np.array([1.0]))
    assert n == 3
    assert all(it.duration_s in (5000.0, 7000.0) for it in out)


def test_resample_falls_back_for_unknown_class():
    items = [_item("a", cpu=9999)]
    samples = {(8000, 1.0): np.array([5000.0])}
    out, n = assign_durations(
        items, "resample", np.random.default_rng(0),
        samples, np.array([4242.0]))
    assert n == 1 and out[0].duration_s == 4242.0


def test_resample_seeds_differ():
    items = [_item(str(i)) for i in range(50)]
    samples = {(8000, 1.0): np.arange(1.0, 200.0)}
    a, _ = assign_durations(
        items, "resample", np.random.default_rng(0), samples,
        samples[(8000, 1.0)])
    b, _ = assign_durations(
        items, "resample", np.random.default_rng(1), samples,
        samples[(8000, 1.0)])
    assert [it.duration_s for it in a] != [it.duration_s for it in b]
