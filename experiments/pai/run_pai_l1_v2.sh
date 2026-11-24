#!/usr/bin/env bash
# PAI-v2020 Layer-1 v2 — core / causal-M / robustness / packed matrices.
#
# Default core matrix is sized for one low-power worker (~7.5-8.5 h on a 4-vCPU server):
#   all 5 domains x all whole-GPU demand blocks x
#   dp, atomic_bestfit, seq_fgd, atomic_fgd, atomic_firstfit x M=stream.
# Causal and robustness stages are separate matrices. A matrix.tsv is written
# before every stage and every line is executed, so no cell is selected by outcome.
#
# Typical two-worker launch (one core each, both read the same prepared blocks):
#   LOW_CORES=0 STAGE=build  bash experiments/pai/run_pai_l1_v2.sh
#   LOW_CORES=0 STAGE=core   bash experiments/pai/run_pai_l1_v2.sh
#   LOW_CORES=1 STAGE=causal bash experiments/pai/run_pai_l1_v2.sh
#
# Useful overrides:
#   STAGE=core|causal|robust|packed|all|build   (default: core)
#   LOW_CORES=0,1        pin mask; use 0 and 1 to run two workers in parallel
#   MAX_BLOCKS=3         first N blocks per domain (calibration / dry checks)
#   DRY_RUN=1            generate matrix_<stage>.tsv, then stop
#   MATRIX_FORCE=1       regenerate an existing matrix file
#   CORE_POLICIES=...    default "dp atomic_bestfit seq_fgd atomic_fgd atomic_firstfit"
#   CAUSAL_POLICIES=...  default "dp seq_fgd atomic_fgd"
#
# Data layout:
#   data/trace-gpu-v2020/blocks_whole_l1v2/          load=1.2, prefix pods 50%
#   data/trace-gpu-v2020/blocks_whole_l1v2_load<L>/  robustness loads
#   results/pai_l1_v2/matrix_<stage>.tsv             pre-registered run list
#   results/pai_l1_v2/<stage>/.../summary.csv        per-cell artifacts
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
STAGE=${STAGE:-core}
DER=${DER:-data/trace-gpu-v2020/blocks_whole_l1v2}
OUT=${OUT:-results/pai_l1_v2}
LOW_CORES=${LOW_CORES:-0,1}
NICE_LEVEL=${NICE_LEVEL:-19}
MAX_BLOCKS=${MAX_BLOCKS:-0}
MATRIX_FORCE=${MATRIX_FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_BUILD=${SKIP_BUILD:-0}

DOMAINS=${DOMAINS:-"MISC P100 T4 V100 V100M32"}
CORE_POLICIES=${CORE_POLICIES:-"dp atomic_bestfit seq_fgd atomic_fgd atomic_firstfit"}
CAUSAL_POLICIES=${CAUSAL_POLICIES:-"dp seq_fgd atomic_fgd"}
ROBUST_POLICIES=${ROBUST_POLICIES:-"dp atomic_bestfit seq_fgd atomic_fgd"}
PACKED_POLICIES=${PACKED_POLICIES:-"bestfit_gang firstfit_gang atomic_firstfit"}
LOADS=${LOADS:-"1.0 1.1 1.3"}
ROBUST_BLOCKS=${ROBUST_BLOCKS:-30}
PACKED_BLOCKS=${PACKED_BLOCKS:-10}
PREFIX_FRAC=${PREFIX_FRAC:-0.5}

LOW=(nice -n "$NICE_LEVEL" taskset -c "$LOW_CORES")

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUT/logs"
MASTER="$OUT/logs/${STAGE}_master.log"
stamp() { date '+%Y-%m-%dT%H:%M:%S'; }
say() { echo "=== $(stamp) $*" | tee -a "$MASTER"; }

prefix_pattern() {
  local pct
  pct=$(printf '%02d' "$("$PY" -c "print(round(float('$PREFIX_FRAC')*100))")")
  echo "$DER/pods_*_prefix${pct}.csv"
}

build_blocks() {
  local der="$1" load="$2" prefix="$3" force="${4:-0}"
  if [ "$SKIP_BUILD" = 1 ]; then
    return 0
  fi
  if [ "$force" != 1 ] && [ -f "$der/blocks.tsv" ]; then
    return 0
  fi
  local lock="$OUT/logs/.build_$(basename "$der").lock"
  (
    flock -x 9
    if [ "$force" != 1 ] && [ -f "$der/blocks.tsv" ]; then
      exit 0
    fi
    say "BUILD $der load=$load prefix=${prefix:-none}"
    local extra=()
    if [ -n "$prefix" ]; then
      extra=(--prefix-fracs "$prefix")
    fi
    "${LOW[@]}" "$PY" scripts/make_pai_blocks.py \
      --domains $DOMAINS --load "$load" "${extra[@]}" --out "$der" \
      >>"$OUT/logs/build_$(basename "$der").log" 2>&1 || {
        say "BUILD FAILED $der"; exit 1;
      }
  ) 9>"$lock"
  return $?
}

ensure_core_blocks() {
  if [ "$SKIP_BUILD" = 1 ]; then
    return 0
  fi
  build_blocks "$DER" 1.2 "$PREFIX_FRAC"
  if ! compgen -G "$(prefix_pattern)" >/dev/null; then
    say "prefix pods missing under $DER; rebuilding with --prefix-fracs $PREFIX_FRAC"
    build_blocks "$DER" 1.2 "$PREFIX_FRAC" 1
  fi
}

limited_blocks() {
  local der="$1" dom="$2" limit="$3"
  if [ "$limit" -gt 0 ]; then
    awk -F'\t' -v d="$dom" '$1==d' "$der/blocks.tsv" | head -n "$limit"
  else
    awk -F'\t' -v d="$dom" '$1==d' "$der/blocks.tsv"
  fi
}

gen_core() {
  local f="$OUT/matrix_core.tsv"
  if [ -f "$f" ] && [ "$MATRIX_FORCE" != 1 ]; then
    say "MATRIX reuse $f"
    return 0
  fi
  : > "$f"
  local dom bi stream pods nodes reached gangs items demand pol
  for dom in $DOMAINS; do
    limited_blocks "$DER" "$dom" "$MAX_BLOCKS" |
    while IFS=$'\t' read -r _d bi stream pods nodes reached gangs items demand; do
      for pol in $CORE_POLICIES; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$dom" "$bi" "$pol" "stream" \
          "$DER/$stream" "$DER/$pods" "$DER/$nodes" \
          "$OUT/core/$dom/block$bi/${pol}__stream" >> "$f"
      done
    done
  done
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

gen_causal() {
  local f="$OUT/matrix_causal.tsv"
  if [ -f "$f" ] && [ "$MATRIX_FORCE" != 1 ]; then
    say "MATRIX reuse $f"
    return 0
  fi
  : > "$f"
  local dom bi stream pods nodes reached gangs items demand pol prev prefix
  for dom in $DOMAINS; do
    prev=""
    limited_blocks "$DER" "$dom" "$MAX_BLOCKS" |
    while IFS=$'\t' read -r _d bi stream pods nodes reached gangs items demand; do
      if [ -n "$prev" ]; then
        for pol in $CAUSAL_POLICIES; do
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dom" "$bi" "$pol" "stale:$DER/$prev" \
            "$DER/$stream" "$DER/$pods" "$DER/$nodes" \
            "$OUT/causal_prev/$dom/block$bi/${pol}__mprev" >> "$f"
        done
      fi
      prev="$pods"
    done
  done
  for dom in $DOMAINS; do
    limited_blocks "$DER" "$dom" "$MAX_BLOCKS" |
    while IFS=$'\t' read -r _d bi stream pods nodes reached gangs items demand; do
      if [ $((10#$bi % 2)) -ne 0 ]; then
        continue
      fi
      prefix="${pods%.csv}_prefix50.csv"
      [ -f "$DER/$prefix" ] || continue
      for pol in $CAUSAL_POLICIES; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$dom" "$bi" "$pol" "stale:$DER/$prefix" \
          "$DER/$stream" "$DER/$pods" "$DER/$nodes" \
          "$OUT/causal_prefix50/$dom/block$bi/${pol}__mpref50" >> "$f"
      done
    done
  done
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

gen_robust() {
  local f="$OUT/matrix_robust.tsv"
  if [ -f "$f" ] && [ "$MATRIX_FORCE" != 1 ]; then
    say "MATRIX reuse $f"
    return 0
  fi
  : > "$f"
  local load der dom bi stream pods nodes reached gangs items demand pol
  for load in $LOADS; do
    der="data/trace-gpu-v2020/blocks_whole_l1v2_load${load}"
    build_blocks "$der" "$load" ""
    for dom in $DOMAINS; do
      limited_blocks "$der" "$dom" "$ROBUST_BLOCKS" |
      while IFS=$'\t' read -r _d bi stream pods nodes reached gangs items demand; do
        for pol in $ROBUST_POLICIES; do
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dom" "$bi" "$pol" "stream" \
            "$der/$stream" "$der/$pods" "$der/$nodes" \
            "$OUT/robust/load${load}/$dom/block$bi/${pol}__stream" >> "$f"
        done
      done
    done
  done
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

gen_packed() {
  local f="$OUT/matrix_packed.tsv"
  if [ -f "$f" ] && [ "$MATRIX_FORCE" != 1 ]; then
    say "MATRIX reuse $f"
    return 0
  fi
  : > "$f"
  local dom bi stream pods nodes reached gangs items demand pol
  for dom in $DOMAINS; do
    limited_blocks "$DER" "$dom" "$PACKED_BLOCKS" |
    while IFS=$'\t' read -r _d bi stream pods nodes reached gangs items demand; do
      for pol in $PACKED_POLICIES; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$dom" "$bi" "$pol" "stream" \
          "$DER/$stream" "$DER/$pods" "$DER/$nodes" \
          "$OUT/packed/$dom/block$bi/${pol}__stream" >> "$f"
      done
    done
  done
  say "MATRIX write $f lines=$(wc -l < "$f")"
}

run_cell() {
  local log="$1"; shift
  say "RUN  $*"
  "${LOW[@]}" "$@" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then say "FAIL rc=$rc  $*"; else say "DONE $*"; fi
  return 0
}

run_matrix() {
  local f="$1"
  local total
  total=$(wc -l < "$f")
  say "STAGE $STAGE matrix=$f cells=$total cores=$LOW_CORES"
  if [ "$DRY_RUN" = 1 ]; then
    say "DRY_RUN stop"
    return 0
  fi
  local dom bi pol mspec stream pods nodes outdir d
  while IFS=$'\t' read -r dom bi pol mspec stream pods nodes outdir; do
    [ -z "${dom:-}" ] && continue
    d="$outdir"
    if [ -f "$d/summary.csv" ]; then
      say "SKIP $d"
      continue
    fi
    mkdir -p "$d"
    run_cell "$d/run.log" "$PY" scripts/calibrate_gang.py --policy "$pol" \
      --streams "$stream" --pods-csv "$pods" --nodes-csv "$nodes" \
      --policy-m "$mspec" --seeds 1 --out "$d"
  done < "$f"
}

case "$STAGE" in
  build)
    ensure_core_blocks
    say "BUILD DONE"
    exit 0
    ;;
  core)
    ensure_core_blocks
    gen_core
    run_matrix "$OUT/matrix_core.tsv"
    ;;
  causal)
    ensure_core_blocks
    gen_causal
    run_matrix "$OUT/matrix_causal.tsv"
    ;;
  robust)
    gen_robust
    run_matrix "$OUT/matrix_robust.tsv"
    ;;
  packed)
    ensure_core_blocks
    gen_packed
    run_matrix "$OUT/matrix_packed.tsv"
    ;;
  all)
    ensure_core_blocks
    gen_core
    run_matrix "$OUT/matrix_core.tsv"
    gen_causal
    run_matrix "$OUT/matrix_causal.tsv"
    gen_robust
    run_matrix "$OUT/matrix_robust.tsv"
    gen_packed
    run_matrix "$OUT/matrix_packed.tsv"
    ;;
  *)
    echo "unknown STAGE=$STAGE (want build|core|causal|robust|packed|all)" >&2
    exit 1
    ;;
esac

say "ALL DONE stage=$STAGE"
