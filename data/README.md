# Data inventory

| path | content | license / source |
|---|---|---|
| `trace-gpu-v2020/` | Alibaba PAI v2020: fetch script, checksums, data card (raw tables are downloaded) | CC BY 4.0; Weng et al., NSDI'22 |
| `trace-gpu-v2023/csv/` | Alibaba cluster-trace-gpu-v2023: 1213 nodes, 6212 GPUs, 8152 pods | research/study use only; Alibaba clusterdata |
| `reference/` | FGD authors' cached results (appendix B check) | Apache-2.0; hkust-adsl simulator |
| `gang/` | small synthetic gang streams for tests and `make smoke` | generated; see `gang/README.md` |

## PAI 2020 raw

`trace-gpu-v2020/raw/` (~329 MB) is not committed:

```bash
bash data/trace-gpu-v2020/fetch.sh
sha256sum -c data/trace-gpu-v2020/SHA256SUMS
```

Three tables are used: `pai_job_table.csv`, `pai_task_table.csv`,
`pai_machine_spec.csv`. `pai_instance_table` is not required.

## Generated, not committed

- `trace-gpu-v2020/blocks_mixed_l1v2/` (~360 MB) — mixed blocks of chapter 3
  (`make blocks-mixed`);
- `trace-gpu-v2020/blocks_mixed_l1v2_flat/` — uniform-M pods (same command);
- `trace-gpu-v2020/blocks_f1_*/` — F1 variants (`make run-f1`, stage build);
- `trace-gpu-v2020/blocks_whole_l1v2/` — whole-GPU-only blocks (`make pai-whole`).

Reconstruction rules and filters: `docs/DATA_CARDS.md`.
