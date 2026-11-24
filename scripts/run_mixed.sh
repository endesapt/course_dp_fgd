#!/usr/bin/env bash
# Mixed PAI blocks: whole + FRACTIONAL gangs and singletons, every item placed by
# the policy under test (--background gang). Pre-registration:
# docs/prereg/PREREG_mixed_gangs.md. Run from any cwd (cd to repo root).
#   STAGE=build                 bash scripts/run_mixed.sh   # blocks + uniform pods
#   STAGE=smoke CORE=1          bash scripts/run_mixed.sh   # 1 block/domain + provenance
#   STAGE=run WORKER=w NWORKERS=n CORE=c  bash scripts/run_mixed.sh
#   STAGE=all  (build -> smoke -> 3 detached workers on cores 1,2,3)
set -u
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-smoke}
WHOLE=${WHOLE:-data/trace-gpu-v2020/blocks_whole_l1v2}
DER=${DER:-data/trace-gpu-v2020/blocks_mixed_l1v2}
FLAT=${FLAT:-data/trace-gpu-v2020/blocks_mixed_l1v2_flat}
OUT=${OUT:-results/pai_mixed}
CORE=${CORE:-1}
WORKER=${WORKER:-0}
NWORKERS=${NWORKERS:-3}
NICE_LEVEL=${NICE_LEVEL:-19}
DOMAINS=${DOMAINS:-"MISC V100 V100M32"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "$STAGE" = smoke ]; then OUT=${OUT}_smoke; MAX_BLOCKS=2; else MAX_BLOCKS=${MAX_BLOCKS:-0}; fi
mkdir -p "$OUT/logs" "$FLAT"
MASTER="$OUT/logs/master_${STAGE}_w${WORKER}.log"
say() { echo "=== $(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$MASTER"; }

# arm -> "policy|mspec-kind" ; priority order (A: main, B: baselines, C: value of M)
ARMS="dp_stream seq_stream abf_stream \
      aff_stream abfgpu_stream adot_stream apack_stream arand_stream \
      dp_prev seq_prev dp_uniform seq_uniform"
arm_policy() {
  case "$1" in
    dp_*) echo dp ;; seq_*) echo seq_fgd ;; abf_*) echo atomic_bestfit ;;
    aff_*) echo atomic_firstfit ;; abfgpu_*) echo atomic_bestfit_gpu ;;
    adot_*) echo atomic_dotprod ;; apack_*) echo atomic_packing ;; arand_*) echo atomic_random ;;
  esac
}

build() {
  if [ ! -f "$DER/blocks.tsv" ]; then
    say "BUILD mixed blocks -> $DER"
    nice -n "$NICE_LEVEL" taskset -c "$CORE" "$PY" scripts/make_pai_blocks.py --mixed \
      --domains $DOMAINS --load 1.2 --out "$DER" >"$OUT/logs/build_blocks.log" 2>&1 \
      || { say "BUILD FAILED (see build_blocks.log)"; exit 1; }
  fi
  local dom f
  for dom in $DOMAINS; do
    f="$FLAT/pods_${dom}_uniform.csv"
    [ -f "$f" ] && continue
    "$PY" - "$DER" "$dom" "$f" <<'EOF'
import sys, glob, pandas as pd
der, dom, out = sys.argv[1:4]
fs = sorted(glob.glob(f"{der}/pods_{dom}_block[0-9][0-9][0-9][0-9].csv"))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
u = df.drop_duplicates(subset=["cpu_milli", "num_gpu", "gpu_milli"]).copy()
u["name"] = [f"uniform:{i}" for i in range(len(u))]
u["creation_time"] = 0.0
u[["name", "creation_time", "cpu_milli", "num_gpu", "gpu_milli", "memory_mib"]].to_csv(out, index=False)
print(f"{dom}: {len(fs)} blocks, {len(u)} uniform classes -> {out}")
EOF
  done
  say "BUILD done: $(wc -l < "$DER/blocks.tsv") blocks"
}

gen_matrix() {
  local f="$OUT/matrix.tsv"
  if [ -f "$f" ]; then say "MATRIX reuse $f"; return 0; fi
  : > "$f"
  local arm pol spec dom bi stream pods nodes rest prev
  for arm in $ARMS; do
    pol=$(arm_policy "$arm")
    for dom in $DOMAINS; do
      prev=""
      awk -F'\t' -v d="$dom" '$1==d' "$DER/blocks.tsv" |
      { if [ "$MAX_BLOCKS" -gt 0 ]; then head -n "$MAX_BLOCKS"; else cat; fi; } |
      while IFS=$'\t' read -r _d bi stream pods nodes rest; do
        case "$arm" in
          # oracle M for mixed blocks = per-worker histogram of THIS block's
          # pods (same construction as prev/uniform). The `stream` spec rebuilds
          # M from synthetic gang shapes and rejects fractional gang workers.
          *_stream)  spec="stale:$DER/$pods" ;;
          *_uniform) spec="stale:$FLAT/pods_${dom}_uniform.csv" ;;
          *_prev)    spec="${prev:+stale:$DER/$prev}" ;;
        esac
        if [ -n "$spec" ]; then
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dom" "$bi" "$pol" "$spec" gang \
            "$DER/$stream" "$DER/$pods" "$DER/$nodes" "$OUT/$arm/$dom/block$bi" >> "$f"
        fi
        prev="$pods"
      done
    done
  done
  if [ "$STAGE" = smoke ]; then
    # provenance: new code on WHOLE blocks must reproduce results/pai_l1_v2/core
    for pol in dp seq_fgd atomic_bestfit; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' V100 0 "$pol" stream fgd_q \
        "$WHOLE/stream_V100_block0000.jsonl" "$WHOLE/pods_V100_block0000.csv" \
        "$WHOLE/nodes_V100.csv" "$OUT/provenance/V100/block0/${pol}" >> "$f"
    done
  fi
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

run_matrix() {
  local f="$OUT/matrix.tsv" n=0 dom bi pol spec bg stream pods nodes d rc t0
  say "START worker=$WORKER/$NWORKERS core=$CORE lines=$(wc -l < "$f")"
  while IFS=$'\t' read -r dom bi pol spec bg stream pods nodes d; do
    n=$((n + 1))
    [ $(( (n - 1) % NWORKERS )) -eq "$WORKER" ] || continue
    [ -f "$d/summary.csv" ] && continue
    mkdir -p "$d"
    t0=$(date +%s)
    nice -n "$NICE_LEVEL" taskset -c "$CORE" "$PY" scripts/calibrate_gang.py --policy "$pol" \
      --background "$bg" --streams "$stream" --pods-csv "$pods" --nodes-csv "$nodes" \
      --policy-m "$spec" --seeds 1 --out "$d" >"$d/run.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then say "FAIL rc=$rc $d"; else say "DONE $(( $(date +%s) - t0 ))s $d"; fi
  done < "$f"
  say "WORKER DONE worker=$WORKER"
}

case "$STAGE" in
  build) build ;;
  smoke) build; gen_matrix; NWORKERS=1; WORKER=0; run_matrix ;;
  run)   gen_matrix; run_matrix ;;
  all)
    build
    STAGE=smoke OUT=${OUT} bash "$0" || exit 1
    if grep -q FAIL "${OUT}_smoke/logs/master_smoke_w0.log"; then
      say "SMOKE has FAIL lines -> not starting full run"; exit 1
    fi
    gen_matrix   # written once here, before any worker starts (no race)
    WORKER=0 CORE=1 STAGE=run bash "$0" &
    WORKER=1 CORE=2 STAGE=run bash "$0" &
    WORKER=2 CORE=3 STAGE=run bash "$0" &
    wait
    say "ALL WORKERS DONE"
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
