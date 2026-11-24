"""Duration assignment for Layer-2 replay.

Modes:
- "field":     items carry real duration_s (spot-gpu streams) — scaled, not
               resampled.
- "empirical": sample per (cpu_milli, gpu_demand) class from a duration table
               (`duration_samples`), falling back to the global pool.
- "constant":  fixed --duration-s.

`scale` (--duration-scale) is the synthetic congestion knob: it multiplies
EVERY duration (real or assigned), raising instantaneous load without touching
arrival times or placement geometry.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..items import StreamItem
from .v2023 import gpu_demand

MIN_DURATION_S = 1e-6


def duration_samples(csv_path) -> tuple[dict, np.ndarray]:
    """Per-class duration samples (s) + global fallback pool.

    Sources: spot job_info_df.csv (`duration` column, cpu_request/gpu_request)
    or v2023 pods (deletion_time - creation_time)."""
    df = pd.read_csv(csv_path)
    if "duration" in df.columns:
        df = df.assign(
            cpu_milli=(df["cpu_request"] * 1000).round().astype(int),
            gpu_demand=np.where(
                df["gpu_request"] >= 1.0, df["gpu_request"].astype(int),
                (df["gpu_request"] * 1000).round().astype(int) / 1000.0,
            ),
            dur=df["duration"].astype(float),
        )
    elif {"deletion_time", "creation_time"} <= set(df.columns):
        df = df.assign(gpu_demand=df.apply(gpu_demand, axis=1))
        df["dur"] = (df["deletion_time"] - df["creation_time"]).astype(float)
    else:
        raise ValueError(f"{csv_path}: no usable duration columns")
    df = df[df["dur"] > 0]
    samples = {k: v.to_numpy() for k, v in df.groupby(["cpu_milli", "gpu_demand"])["dur"]}
    return samples, df["dur"].to_numpy()


def durations_from_file(path) -> dict:
    """{(cpu_milli, gpu_demand): [duration_s]} from a stream .jsonl (item
    duration_s field) or a durations .csv (same dispatch as duration_samples).
    Source for deployable occupancy weight tables (PAST-window data only)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        out: dict = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            it = StreamItem.from_dict(json.loads(line))
            if it.duration_s is not None and it.duration_s > 0:
                out.setdefault(it.class_key, []).append(float(it.duration_s))
        if not out:
            raise ValueError(f"{path}: stream items carry no duration_s")
        return out
    samples, _ = duration_samples(path)
    return {k: v.tolist() for k, v in samples.items()}


def duration_means(path) -> dict:
    """{class_key: mean duration_s} — the E[L_m] table for occupancy-
    weighted estimators."""
    return {k: float(np.mean(v)) for k, v in durations_from_file(path).items()}


def assign_durations(
    items,
    mode: str,
    rng: np.random.Generator,
    samples: dict | None = None,
    fallback: np.ndarray | None = None,
    constant_s: float | None = None,
    scale: float = 1.0,
) -> tuple[list[StreamItem], int]:
    """Return (new items with duration_s stamped, count newly assigned).

    Items already carrying duration_s keep their real value scaled by `scale`
    (spot-gpu streams); the rest draw per `mode`. mode="field" asserts that
    every item already has duration_s (scaled only). mode="resample" redraws
    EVERY item's duration from the class-conditional pool, ignoring stamped
    values — use it for real run-seed variance on duration-carrying streams
    ("empirical" alone only fills items missing duration_s, i.e. it is a
    no-op on spot-gpu streams)."""
    if mode not in ("field", "empirical", "constant", "resample"):
        raise ValueError(f"unknown durations mode {mode!r}")
    out: list[StreamItem] = []
    n_assigned = 0
    for it in items:
        if it.duration_s is not None and mode != "resample":
            out.append(dataclasses.replace(
                it, duration_s=max(float(it.duration_s) * scale, MIN_DURATION_S)))
            continue
        if mode == "field":
            raise ValueError(
                f"durations='field' but item {it.gang_id} has no duration_s — "
                "use empirical/constant for streams without durations"
            )
        if mode == "constant":
            d = float(constant_s)
        elif mode == "resample":
            if samples is None or fallback is None:
                raise ValueError("durations='resample' requires duration samples")
            pool = samples.get(it.class_key)
            d = float(rng.choice(pool if pool is not None and len(pool) else fallback))
        else:  # empirical
            if samples is None or fallback is None:
                raise ValueError("durations='empirical' requires duration samples")
            pool = samples.get(it.class_key)
            d = float(rng.choice(pool if pool is not None and len(pool) else fallback))
        out.append(dataclasses.replace(it, duration_s=max(d * scale, MIN_DURATION_S)))
        n_assigned += 1
    return out, n_assigned
