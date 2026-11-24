# `data/trace-gpu-v2020/` — Alibaba PAI v2020 full-replay adapter

Source: [alibaba/clusterdata → cluster-trace-gpu-v2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020)
(CC BY 4.0; cite Weng et al., NSDI'22 “MLaaS in the Wild”).

The released CSVs are **headerless**; the sibling `*.header` files name the
columns. Download and extract:

```bash
mkdir -p data/trace-gpu-v2020/raw && cd data/trace-gpu-v2020/raw
for f in pai_job_table pai_task_table pai_machine_spec; do
  curl -O "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/$f.tar.gz"
  tar xzf "$f.tar.gz"
done
```

`pai_instance_table.tar.gz` (663 MB) is **not required** for the current loader.
It is the only table with real `instance → machine` placement and is reserved
for a later placement-validation pass.

## Files checked into the adapter

| file | role |
|---|---|
| `README.md` | this data card |
| `raw/pai_{job,task,machine_spec}.csv` | downloaded tables (gitignored; not committed) |
| `derived/` | generated nodes/pods/streams/manifest (regenerate on demand) |

Verified checksums (2026-11-02):

```
5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0  pai_job_table.tar.gz
cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d  pai_machine_spec.tar.gz
cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5  pai_task_table.tar.gz
379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c  pai_job_table.csv
251e9e33649155a6a935fdc1dc0e90cd4698206936b60bd8fce60ff2ccd51439  pai_machine_spec.csv
6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de  pai_task_table.csv
```

## Tables and reconstruction rules

`pai_job_table`: `job_name, inst_id, user, status, start_time, end_time`.
`start_time` is the **job submission** time.

`pai_task_table`: `job_name, task_name, inst_num, status, start_time, end_time,
plan_cpu, plan_mem, plan_gpu, gpu_type`.

`pai_machine_spec`: `machine, gpu_type, cap_cpu, cap_mem, cap_gpu`.

The loader (`src/fgdg2/datasets/pai.py`) applies these rules:

- **one task = one gang**; `inst_num = k` workers, homogeneous per-worker demand;
- **arrival** = `job.start_time` (submission), not `task.start_time`; the gap
  `task.start_time − job.start_time` is the observed wait the simulator should
  reproduce;
- **duration** = `task.end_time − task.start_time`; missing/zero durations are
  omitted from Layer-1 and filtered before Layer-2;
- **GPU demand** = `plan_gpu / 100` (PAI stores GPU requests as percent);
- **whole-GPU worker** iff `plan_gpu >= 100` and `plan_gpu % 100 == 0`;
- **fractional worker** iff `0 < plan_gpu < 100`;
- combined requests (`plan_gpu = 150`, `250`) are outside the FGD model and are
  dropped;
- **fractional gangs** (`inst_num > 1` with fractional workers) are outside the
  whole-worker gang layer and are dropped; fractional **singletons** stay as
  background;
- **CPU** = `plan_cpu * 10` milli-cores (`plan_cpu` is percent; `600` → `6000`
  milli = 6 cores); node CPU = `cap_cpu * 1000`;
- **memory** = `plan_mem * 1024` MiB; node memory = `cap_mem * 1024` MiB
  (memory is not part of the FGD class key and is inert in the current runner);
- **`gpu_type` is a hard domain constraint** (the trace records the assigned
  type, not a separate requested type). Replay is per-domain, exactly like the
  spot-gpu adapter. Tasks whose whole-worker demand exceeds the largest node of
  their domain are dropped (`P100`/`T4` max 2 GPUs; `MISC`/`V100`/`V100M32`
  max 8).

## Population counts (full trace)

| item | count |
|---|---:|
| jobs | 1,055,501 |
| tasks | 1,261,050 |
| whole-GPU tasks (`plan_gpu ∈ {100,200,…,800}`) | 472,905 |
| whole-GPU multi-instance tasks (gangs) | 74,068 |
| whole-GPU worker instances | 1,324,282 |
| fractional tasks | 564,145 |
| fractional multi-instance tasks (dropped) | 139,186 |
| fractional singletons (kept as background) | 424,959 |
| zero/NaN-GPU tasks (dropped) | 223,965 |
| unsupported combined requests (dropped) | 35 |

Machine inventory: 1,897 machines, **6,742 GPUs**:

| `gpu_type` | nodes | GPUs | max GPU/node |
|---|---:|---:|---:|
| MISC | 280 | 2,240 | 8 |
| P100 | 798 | 1,596 | 2 |
| T4 | 497 | 994 | 2 |
| V100 | 104 | 832 | 8 |
| V100M32 | 135 | 1,080 | 8 |
| CPU | 83 | 0 | 0 |

Per-domain replayable tasks after filters (see `manifest.json` for exact
counts, drops and windows):

| domain | tasks kept | whole gangs | capacity |
|---|---:|---:|---:|
| MISC | 618,210 | 34,288 | 2,240 |
| P100 | 62,427 | 14,746 | 1,596 |
| T4 | 172,640 | 14,827 | 994 |
| V100 | 26,699 | 4,900 | 832 |
| V100M32 | 17,558 | 5,262 | 1,080 |

`kept` includes whole-GPU tasks plus fractional singletons; it excludes
fractional gangs and the other drop classes above.

## Legacy `PAI_GANG_SHAPES` relationship

`src/fgdg2/datasets/synth.py` contains a separate, legacy `PAI_GANG_SHAPES`
mix (9 shapes, 4,015 tasks) used only by the synthetic v2023 gang streams.
That constant is **not** the full-replay population above: the released trace
has 74,068 whole-GPU multi-instance tasks, and the full replay uses the
observed arrivals directly. Do not treat the synthetic stream weights and the
per-domain stream shape histogram as the same population, and do not “fix” one
to match the other without re-deriving the synthetic artifact.

## Generate derived streams

```bash
# Layer 1 (fill-up): demand-capped stream per GPU-type domain
python scripts/make_pai_streams.py --load 2.0 --out data/trace-gpu-v2020/derived

# Layer 2 (temporal): peak 7-day window per domain
python scripts/make_pai_streams.py --layer2 --days 7 \
    --out data/trace-gpu-v2020/derived

# Layer 2 with an explicit window in DAYS (PAI time is offset but days align)
python scripts/make_pai_streams.py --layer2 --window-range 10:17 \
    --out data/trace-gpu-v2020/derived
```

Each domain writes `nodes_<T>.csv`, `pods_<T>.csv` (one row per worker
instance), `stream_<T>_*.jsonl`, and a shared `manifest*.json`.

## Honesty notes

- This is **reconstructed** production structure. The trace has no
  gang atomicity flag; `inst_num > 1` is the task's co-scheduled instance
  count, which is the closest available gang signal.
- `gpu_type` is the assigned type, not a recorded request. Treating it as a
  hard domain is an assumption, documented above.
- No migration, checkpoint, or preemption-cost fields exist in PAI v2020.
- PAI has a different GPU mix and much larger gangs than the v2023/spot-gpu
  traces. A “does not transfer” result is a valid cross-trace result.
- The optional `pai_instance_table` is the only way to validate the
  reconstruction against real `instance → machine` assignments; that check is
  not part of the current loader.
