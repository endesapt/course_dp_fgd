# `data/gang/` — gang-phase dataset (schema v2, 2026-09-16; see docs/FINDINGS.md §3)

Three things live here, with different epistemic status — never conflate them:

| artifact | status | used for |
|---|---|---|
| `trace_coarrival_stats.json` | **measured from the real trace** | honesty layer: the v2023 trace has no verified gang structure |
| worker classes (whole and fractional) | **real, recomputed from the v2023 trace on demand** | sampling + the FGD objective `F_n(M)` |
| default gang-shape mix | **real, measured from Alibaba PAI v2020** | the distribution over (k workers × d GPU/worker) |
| `synth_gangs_seed0_*.jsonl` | **synthetic experiment layer** | arrival streams for the gang policies |

## Provenance and license

- v2023 trace: `data/trace-gpu-v2023/csv/openb_pod_list_default.csv` (Alibaba
  `cluster-trace-gpu-v2023`; **research/study use only**, MD5
  `39D1F424A28C017DA9ECFD0C24200924`; nodes `openb_node_list_gpu_node.csv`,
  capacity 6212 GPUs / 1213 nodes). All files here inherit these terms.
- default gang shapes: Alibaba **PAI v2020** trace (`cluster-trace-gpu-v2020`,
  **CC BY 4.0**, cite Weng et al., NSDI'22 "MLaaS in the Wild"), derived from the
  4,015 multi-instance whole-worker tasks (see the table below).

No `m_whole.json`/`m_frac.json`: both M's are recomputed from the pod CSV by
`build_m_whole` / `build_m_fractional`; stream headers carry their sha256
fingerprints and the runner hard-fails on mismatch.

## M_whole and M_frac (measured)

- whole-GPU pods (`num_gpu >= 1 and gpu_milli == 1000`): **3986 / 8152**, 71.5% of
  trace GPU demand. **42 classes, kept mass 1.0** (no 95% cut — that cut is a
  calibration-side reference-fidelity device and would drop every multi-GPU class):
  demands **{1, 2, 4, 8}**, multi-GPU workers = 10.2% of whole demand, E[d] = 1.093.
- fractional pods (`num_gpu == 1 and 0 < gpu_milli < 1000`): **3078 pods**, 28.5% of
  trace GPU demand. **39 classes**, E[d] = 0.563 (e.g. 460/500/810 milli).
- CPU-only pods are out of scope (zero GPU demand).

## Default gang-shape mix (real, from PAI v2020)

Joint distribution of the 4,015 whole-worker multi-instance tasks
(task-count weights; shape `(2,3)` dropped — 2 tasks, no 3-GPU class exists in
v2023 — and weights renormalized):

| shape | tasks | weight | shape | tasks | weight |
|---|---|---|---|---|---|
| 2×4 | 1331 | 0.3316 | 2×2 | 194 | 0.0483 |
| 2×1 | 1122 | 0.2796 | 4×1 | 83 | 0.0207 |
| 4×2 | 1047 | 0.2609 | 3×1 | 16 | 0.0040 |
| 8×1 | 213 | 0.0531 | 5×1 / 6×1 | 7 | 0.0017 |

(`P × Q` = P instances × Q GPUs per instance.) The mix is a **knob**:
`--gang-shapes "2x4:0.4,4x1:0.3"` overrides it.

## Synthetic streams (experiment layer, schema v2)

Per arrival, a group is drawn (probability set so the EXPECTED demand shares match
the targets): **gang** (share `gang_fraction`; shape from the mix; k copies of one
class conditioned on the shape's d), **fractional singleton** (share
`fractional_share`; class from M_frac) or **whole singleton** (the rest; class from
M_whole). Group probabilities `p ∝ share / mean_demand` normalize to the targets;
the legacy single-shape case reduces to `p = f/(k − f·(k−1))`. Stop at
`cumulative demand >= load × capacity` (last arrival may overshoot).

Committed demos (seed 0, load 1.0):

| file | gang shapes (realized) | gang share | frac share |
|---|---|---|---|
| `synth_gangs_seed0_f00_mix_load1_frac28.jsonl` | — (no gangs) | 0.000 | 0.285 |
| `synth_gangs_seed0_f25_mix_load1_frac28.jsonl` | 7 shapes (2×1…8×1 incl. 2×4) | 0.244 | 0.284 |
| `synth_gangs_seed0_f50_mix_load1_frac28.jsonl` | 8 shapes | 0.491 | 0.289 |
| `synth_gangs_seed0_f50_mix_load1_frac50.jsonl` | 7 shapes | 0.501 | 0.499 |
| `synth_gangs_seed0_f50_k16_load1.jsonl` | 16×1 (+16×8) legacy fixed-k | 0.502 | 0.000 |
| `synth_gangs_seed0_f50_k16_load1_g50.jsonl` | 16×d, multi-GPU tilt g=0.5 | 0.491 | 0.000 |

Tolerance bounds (10-seed maxima): `|realized share − target| ≤ 0.026`, arrival
class TV vs `policy_workload` ≤ 0.042; tests add margin (0.04 / 0.06).

Full matrix on demand (not committed):

```
.venv/bin/python scripts/make_gang_dataset.py --full --out data/gang/raw
.venv/bin/python scripts/make_gang_dataset.py --fractions 25 50 --shapes "2x4:0.4,4x1:0.3,8x1:0.3" --fractional-share 0.5
```

### Stream schema (v2)

Header: `type, schema_version=2, git_sha, seed, config {gang_fraction, gang_size
(null when shapes), gang_shapes [{k,d,weight}] or null, load, target_demand,
capacity, multi_gpu_share, fractional_share}, m_whole_sha256,
m_fractional_sha256, source_pod_csv, source_pod_md5, note, realized {n_items,
n_gangs, n_singletons, n_fractional_singletons, realized_gang_demand_share,
realized_fractional_demand_share, realized_shape_counts}`.
Item: `gang_id, gang_size, worker_cpu_milli, worker_gpu_demand, arrival_index,
is_gang` (fractional singleton: `gang_size=1, worker_gpu_demand < 1`).

## Co-arrival honesty layer (measured)

Grouping whole-GPU pods by `(creation_time, cpu_milli, num_gpu)`: 3923 groups —
3864×1, 55×2, 4×3; **3.06% of whole pods in groups > 1**. Same-second co-arrival
is a LOWER bound on real gang structure; the data shows no verified production
gangs — hence the synthetic layer, and the ban on phrasing these gangs as observed.

Regenerate everything (deterministic):

```
.venv/bin/python scripts/make_gang_dataset.py
```
