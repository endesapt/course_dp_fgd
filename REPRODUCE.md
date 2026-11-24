# Воспроизведение

Все команды — из корня репозитория. Интерпретатор задаётся `PY`
(по умолчанию `python3` в Makefile и `.venv/bin/python` в `scripts/*.sh`).

## 0. Окружение и тесты

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
make test        # быстрые тесты (~1 мин)
make test-all    # вместе с медленными
make smoke       # одна маленькая ганг-задача end-to-end
```

## 1. Главы 3–4 из готовых агрегатов (без сырой трассы)

```bash
make figs-mixed   # claims.csv (N01–N55), results/figures/fig4_*, results/tables/tab3_*, tab4_*
make latex-sync   # копия в ../latex_coursework/{images,tables}
```

Входы — только закоммиченные агрегаты:
`results/aggregates/mixed/`, `results/aggregates/f1/`,
`results/aggregates/pai_l1_v2/pai_l1_v2_all_cells.csv` (старые блоки с целыми групповыми задачами, контроль P100/T4).

## 2. Трасса PAI 2020

```bash
bash data/trace-gpu-v2020/fetch.sh
sha256sum -c data/trace-gpu-v2020/SHA256SUMS
make pai-trace-stats   # приложение A: aggregates/mixed/pai_trace_stats.csv, tables/tabA_pai.tex
```

## 3. Блоки и прогоны (часы; рассчитано на Linux, 4 ядра)

Скрипты пишут план прогона `matrix.tsv` до запуска и выполняют каждую
строку; ячейка с готовым `summary.csv` пропускается, поэтому прерванный
прогон можно перезапустить той же командой. Процессы идут с `nice -n 19` и
`taskset` (ядро задаётся `CORE`).

```bash
# 3.1 старые блоки с целыми групповыми задачами + их прогон (контроль
#     P100/T4 и проверка-совпадение в smoke-стадиях ниже)
make pai-whole

# 3.2 смешанные блоки главы 3 (778 блоков, ~360 МБ)
make blocks-mixed
#   то же вручную:
#   python scripts/make_pai_blocks.py --mixed --load 1.2 \
#       --domains MISC V100 V100M32 --out data/trace-gpu-v2020/blocks_mixed_l1v2

# 3.3 основной прогон главы 4: 12 вариантов x 778 блоков = 9 330 ячеек
make run-mixed          # build -> smoke (2 блока/домен) -> 3 воркера
#   или по частям: STAGE=smoke bash scripts/run_mixed.sh
#                  STAGE=run WORKER=0 NWORKERS=3 CORE=1 bash scripts/run_mixed.sh

# 3.4 F1 (раздел 4.2.1): ганги расщеплены в одиночные, спрос сохранён
make run-f1             # нужны результаты 3.3 (сравнение smoke-ячейки)

# 3.5 агрегаты из сырых прогонов
make mixed-aggregates   # MIXED_RUNS=results/pai_mixed MIXED_BLOCKS=data/trace-gpu-v2020/blocks_mixed_l1v2
make f1-aggregates      # F1_RUNS=results/pai_f1
make figs-mixed
```

Одна ячейка (блок × политика × M) вручную:

```bash
B=data/trace-gpu-v2020/blocks_mixed_l1v2
python scripts/calibrate_gang.py --policy dp --background gang \
    --streams $B/stream_MISC_block0000.jsonl \
    --pods-csv $B/pods_MISC_block0000.csv --nodes-csv $B/nodes_MISC.csv \
    --policy-m stale:$B/pods_MISC_block0000.csv --seeds 1 --out results/cell_demo
```

Ожидаемо: `unalloc@100=11.79%`, `gangs=247 placed=170`; столбцы времени
зависят от машины.

Варианты M в `run_mixed.sh`: `*_stream` — гистограмма воркеров этого же блока
(оракул), `*_prev` — прошлого блока, `*_uniform` — все классы домена поровну.

## 4. Приложение B: проверка FGD на v2023

```bash
make v2023   # results/experiments/calib_*.json, results/reference_compare.csv
```

## Проверка этой версии (2026-11-22)

Код `src/` и `tests/` взят с ветки `fractional-gangs` симулятора (коммит
`39fc8bd`; та же версия, что выполняла прогоны на сервере, сверено по md5).
С ним:

- `make test`: все тесты проходят (2 пропущены: нет сырой трассы PAI и старого
  пакета для сравнения);
- `make figs-mixed`, `make mixed-aggregates`, `make f1-aggregates`,
  `make pai-trace-stats` дают claims.csv, агрегаты и таблицы, совпадающие бит
  в бит с использованными в тексте;
- ячейка из раздела 3 совпадает с серверным прогоном во всех столбцах, кроме
  времени.

Рисунки PNG/PDF зависят от версии matplotlib (размер может отличаться на
1–3 пикселя); числа на них берутся из тех же агрегатов.
