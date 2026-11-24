# results/

```
claims.csv                       реестр чисел текста: строки N01–N55 (N40–N53 — F1, N54–N55 — контроль P100/T4)
aggregates/mixed/                глава 4, смешанные блоки PAI (778 блоков)
  mixed_cells.csv                9 330 ячеек: вариант x домен x блок, unalloc@100
  mixed_blocks.csv               состав нагрузки по блокам
  mixed_domain_means.csv         средние по доменам
  mixed_contrasts.csv            парные разницы, повторные выборки по группам блоков, Holm
  mixed_m_levels.csv             цена M (оракул / прошлый блок / равномерный)
  pai_trace_stats.csv            состав трассы (приложение A)
aggregates/f1/                   раздел 4.2.1: f1_cells.csv, f1_contrasts.csv
aggregates/pai_l1_v2/pai_l1_v2_all_cells.csv   старые блоки с целыми групповыми задачами, дробные одиночные ставит FGD (контроль P100/T4)
experiments/, reference_compare.csv            проверка FGD на v2023 (приложение B)
figures/                         fig4_{mixed_domains,mixed_split,m_cost}
tables/                          tab3_blocks, tab4_* (tab4_f1 — раздел 4.2.1), tabA_pai
```

Строки `C*` в `claims.csv` относятся к прежней версии текста и в
текущем тексте не используются. Строки N23–N27 (сравнение со старыми
блоками на 471 блоке) в тексте тоже больше не цитируются.

## Правило чисел

Числа в текст не набираются вручную: источник — `claims.csv`
(`claim_id`, `value`, `source_file`, `filter`, `used_in`). При изменении
агрегата: `make figs-mixed`, затем `make latex-sync`.

## Не коммитится

Сырые прогоны (`results/pai_mixed/`, `results/pai_f1/`, `results/pai_l1_v2/`,
по `summary.csv` и артефакту на ячейку) — восстанавливаются скриптами из
`scripts/` (см. `REPRODUCE.md`).
