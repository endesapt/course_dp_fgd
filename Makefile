PY ?= python3
export PYTHONPATH := src

# All paths are relative to the repository root. Raw PAI tables, derived blocks
# and per-cell run artifacts are not committed (see REPRODUCE.md).
PAI_RAW      ?= data/trace-gpu-v2020/raw
MIXED_BLOCKS ?= data/trace-gpu-v2020/blocks_mixed_l1v2
MIXED_RUNS   ?= results/pai_mixed
F1_RUNS      ?= results/pai_f1
FGDG2_SRC    ?= src

.PHONY: test test-all lint smoke figs-mixed latex-sync mixed-aggregates f1-aggregates \
        pai-trace-stats blocks-mixed run-mixed run-f1 pai-whole v2023 all clean

# --- checks -----------------------------------------------------------------
test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

smoke:
	$(PY) scripts/smoke.py

# --- chapters 3-4 from committed aggregates (no raw data needed) -------------
# stats + claims N01-N55 + figures fig4_* + tables tab3_/tab4_*
figs-mixed:
	$(PY) scripts/make_mixed_stats.py
	$(PY) scripts/make_mixed_figs.py
	$(PY) scripts/make_mixed_tables.py

latex-sync:
	$(PY) scripts/sync_latex.py

# --- rebuild aggregates from raw runs / raw trace ----------------------------
mixed-aggregates:
	$(PY) scripts/collect_mixed_cells.py --runs $(MIXED_RUNS) --blocks $(MIXED_BLOCKS)

f1-aggregates:
	$(PY) scripts/collect_f1_cells.py --runs $(F1_RUNS)

# appendix A (claims N38-N39) + tabA_pai.tex
pai-trace-stats:
	FGDG2_SRC=$(FGDG2_SRC) $(PY) scripts/pai_trace_stats.py --raw $(PAI_RAW)

# --- heavy runs (hours; need data/trace-gpu-v2020/raw) -----------------------
# mixed blocks of chapter 3 (load 1.2, domains MISC V100 V100M32)
blocks-mixed:
	PY=$(PY) STAGE=build bash scripts/run_mixed.sh

# chapter 4: 12 arms x 778 mixed blocks (build -> smoke -> 3 workers)
run-mixed:
	PY=$(PY) STAGE=all bash scripts/run_mixed.sh

# section 4.2.1 (F1): demand-preserving composition variants; needs run-mixed
run-f1:
	PY=$(PY) STAGE=all bash scripts/run_f1.sh

# blocks with whole-GPU gangs only, fractional singletons placed by FGD background
# (P100/T4 control in section 4.4; provenance cells for the smoke stages)
pai-whole:
	PY=$(PY) STAGE=build bash experiments/pai/run_pai_l1_v2.sh
	PY=$(PY) STAGE=core bash experiments/pai/run_pai_l1_v2.sh

# appendix B: FGD check on Alibaba v2023
v2023:
	for pol in fgd_q bestfit packing dotprod clustering random; do \
		$(PY) scripts/calibrate.py --policy $$pol --seeds 4 --fast \
			--out results/experiments || exit 1; \
	done
	$(PY) scripts/compare_reference.py

all: test figs-mixed latex-sync

clean:
	rm -rf results/smoke results/gang_runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
