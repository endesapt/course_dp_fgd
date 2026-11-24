#!/usr/bin/env bash
# F1: demand-preserving composition variants of the mixed PAI blocks
# (split fractional gangs / whole gangs / all gangs into singletons; FGD
# background for fractional singletons). Pre-registration:
# docs/prereg/PREREG_f1_composition.md. Run from any cwd (cd to repo root).
#   STAGE=build                 bash scripts/run_f1.sh   # derive variant blocks
#   STAGE=smoke CORE=1          bash scripts/run_f1.sh   # 2 blocks/domain + provenance + check
#   STAGE=check                 bash scripts/run_f1.sh   # provenance comparison only
#   STAGE=run WORKER=w NWORKERS=n CORE=c  bash scripts/run_f1.sh
#   STAGE=all  (build -> smoke -> 3 detached workers on cores 1,2,3)
set -u
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-smoke}
WHOLE=${WHOLE:-data/trace-gpu-v2020/blocks_whole_l1v2}
MIXED=${MIXED:-data/trace-gpu-v2020/blocks_mixed_l1v2}
ROOT=${ROOT:-data/trace-gpu-v2020}
OUT=${OUT:-results/pai_f1}
CORE=${CORE:-1}
WORKER=${WORKER:-0}
NWORKERS=${NWORKERS:-3}
NICE_LEVEL=${NICE_LEVEL:-19}
DOMAINS=${DOMAINS:-"MISC V100 V100M32"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ "$STAGE" = smoke ]; then OUT=${OUT}_smoke; MAX_BLOCKS=2; else MAX_BLOCKS=${MAX_BLOCKS:-0}; fi
mkdir -p "$OUT/logs"
MASTER="$OUT/logs/master_${STAGE}_w${WORKER}.log"
say() { echo "=== $(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$MASTER"; }

# arm = <variant>__<policy-tag>; priority order: H1, H2, H3, sanity
ARMS="sfg__dp sfg__seq sfg__abf \
      swg__dp swg__seq swg__abf \
      sfgbg__dp sfgbg__seq sfgbg__abf \
      sall__dp sall__seq sall__abf"
arm_policy() {
  case "${1##*__}" in dp) echo dp ;; seq) echo seq_fgd ;; abf) echo atomic_bestfit ;; esac
}
arm_variant() {
  case "${1%%__*}" in
    sfg|sfgbg) echo splitfracgang ;; swg) echo splitwholegang ;; sall) echo splitall ;;
  esac
}
arm_background() { case "${1%%__*}" in sfgbg) echo fgd_q ;; *) echo gang ;; esac; }

build() {
  local v need=""
  for v in splitfracgang splitwholegang splitall; do
    [ -f "$ROOT/blocks_f1_$v/blocks.tsv" ] || need="$need $v"
  done
  if [ -n "$need" ]; then
    say "BUILD variants:$need"
    nice -n "$NICE_LEVEL" taskset -c "$CORE" "$PY" scripts/derive_variants.py \
      --src "$MIXED" --out-root "$ROOT" --variants $need --domains $DOMAINS \
      >"$OUT/logs/build_variants.log" 2>&1 || { say "BUILD FAILED (see build_variants.log)"; exit 1; }
  fi
  for v in splitfracgang splitwholegang splitall; do
    say "BUILD $v: $(wc -l < "$ROOT/blocks_f1_$v/blocks.tsv") blocks"
  done
}

gen_matrix() {
  local f="$OUT/matrix.tsv"
  if [ -f "$f" ]; then say "MATRIX reuse $f"; return 0; fi
  : > "$f"
  local arm pol var bg der dom bi stream pods nodes rest
  for arm in $ARMS; do
    pol=$(arm_policy "$arm"); var=$(arm_variant "$arm"); bg=$(arm_background "$arm")
    der="$ROOT/blocks_f1_$var"
    for dom in $DOMAINS; do
      awk -F'\t' -v d="$dom" '$1==d' "$der/blocks.tsv" |
      { if [ "$MAX_BLOCKS" -gt 0 ]; then head -n "$MAX_BLOCKS"; else cat; fi; } |
      while IFS=$'\t' read -r _d bi stream pods nodes rest; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dom" "$bi" "$pol" "stale:$der/$pods" "$bg" \
          "$der/$stream" "$der/$pods" "$der/$nodes" "$OUT/$arm/$dom/block$bi" >> "$f"
      done
    done
  done
  if [ "$STAGE" = smoke ]; then
    # provenance 1: whole V100 block0 must reproduce results/pai_l1_v2/core
    for pol in dp seq_fgd atomic_bestfit; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' V100 0 "$pol" stream fgd_q \
        "$WHOLE/stream_V100_block0000.jsonl" "$WHOLE/pods_V100_block0000.csv" \
        "$WHOLE/nodes_V100.csv" "$OUT/provenance/whole_V100_block0/${pol}" >> "$f"
    done
    # provenance 2: ORIGINAL mixed MISC block0, dp, must reproduce results/pai_mixed
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' MISC 0 dp "stale:$MIXED/pods_MISC_block0000.csv" gang \
      "$MIXED/stream_MISC_block0000.jsonl" "$MIXED/pods_MISC_block0000.csv" \
      "$MIXED/nodes_MISC.csv" "$OUT/provenance/mixed_MISC_block0/dp" >> "$f"
  fi
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

check_provenance() {
  local base="${OUT%_smoke}_smoke/provenance"
  "$PY" - "$base" <<'EOF' | tee -a "$MASTER"
import csv, sys
from pathlib import Path
base = Path(sys.argv[1])
SKIP = {"wall_time_s", "decide_mean_ms", "decide_p99_ms"}
pairs = [(base / f"whole_V100_block0/{p}/summary.csv",
          Path(f"results/pai_l1_v2/core/V100/block0/{p}__stream/summary.csv"))
         for p in ("dp", "seq_fgd", "atomic_bestfit")]
pairs.append((base / "mixed_MISC_block0/dp/summary.csv",
              Path("results/pai_mixed/dp_stream/MISC/block0/summary.csv")))
bad = 0
for new, ref in pairs:
    if not new.exists() or not ref.exists():
        print(f"PROVENANCE MISSING {new} vs {ref}"); bad += 1; continue
    a = list(csv.DictReader(new.open())); b = list(csv.DictReader(ref.open()))
    diff = [k for k in a[-1] if k not in SKIP and a[-1].get(k) != b[-1].get(k)]
    print(("PROVENANCE OK " if not diff else f"PROVENANCE MISMATCH {diff} ") + str(new))
    bad += bool(diff)
sys.exit(1 if bad else 0)
EOF
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
  check) check_provenance ;;
  smoke)
    build; gen_matrix; NWORKERS=1; WORKER=0; run_matrix
    check_provenance || say "SMOKE PROVENANCE FAIL"
    ;;
  run)   gen_matrix; run_matrix ;;
  all)
    build
    STAGE=smoke OUT=${OUT} bash "$0" || exit 1
    if grep -q "FAIL" "${OUT}_smoke/logs/master_smoke_w0.log"; then
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
