#!/usr/bin/env python3
"""LaTeX tables (need \\usepackage{adjustbox}) for chapter 4 (mixed PAI blocks) from committed aggregates.

Reads results/aggregates/mixed/{mixed_cells,mixed_contrasts}.csv (run
make_mixed_stats.py first). Writes results/tables/tab4_mixed_*.tex;
`make latex-sync` copies them into the LaTeX project.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "results" / "aggregates" / "mixed"
OUT = ROOT / "results" / "tables"
SIG = ["MISC", "V100", "V100M32"]
SCOPES = SIG + ["ALL"]
NAME = {"ALL": "все блоки", "MISC": "MISC", "V100": "V100", "V100M32": "V100M32"}


def num(x, d=2, sign=True):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "\\(-\\)").replace("+", "\\(+\\)") if sign else s


def ci(r):
    return f"{num(r.mean_pp)} [{num(r.ci_lo)}; {num(r.ci_hi)}]"


def wtl(r):
    return f"{r.wins}/{r.ties}/{r.losses}"


def table(name, label, caption, colspec, header, rows, note=None):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{adjustbox}{max width=\\textwidth}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\",
             "\\midrule"]
    for i, r in enumerate(rows):
        if r == "MID":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines += ["\\par\\smallskip", f"\\parbox{{0.95\\textwidth}}{{\\footnotesize {note}}}"]
    lines.append("\\end{table}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def main() -> int:
    cells = pd.read_csv(MIX / "mixed_cells.csv")
    res = pd.read_csv(MIX / "mixed_contrasts.csv")

    def get(label, scope):
        return res[(res.contrast == label) & (res.scope == scope)].iloc[0]

    wide = cells.pivot_table(index=["domain", "block"], columns="arm", values="unalloc100")
    trio = wide[["abf_stream", "seq_stream", "dp_stream"]].dropna()

    rows = []
    for s in SCOPES:
        t = trio if s == "ALL" else trio[trio.index.get_level_values(0) == s]
        r = get("DP - aBF", s)
        if s == "ALL":
            rows.append("MID")
        rows.append([NAME[s], str(len(t))] + [num(t[a].mean(), sign=False)
                                              for a in ("abf_stream", "seq_stream", "dp_stream")]
                    + [ci(r), wtl(r)])
    table("tab4_mixed_main", "tab:mixedmain",
          "Сравнение политик по доменам",
          "lrrrrll",
          "Домен & Блоков & BestFit & FGD-посл. & DP-FGD & DP-FGD \\(-\\) BestFit, п.п. [95\\% ДИ] & Л/Р/Х",
          rows,
          "\\(U_{100}\\) в \\% ёмкости, средние по блокам, где есть все три политики. "
          "Л/Р/Х: число блоков, где DP-FGD лучше, равен и хуже BestFit. В скобках 95\\% ДИ.")

    rows = []
    for s in SCOPES:
        a, b, c = get("DP - aBF", s), get("seq - aBF", s), get("DP - seq", s)
        if s == "ALL":
            rows.append("MID")
        rows.append([NAME[s], ci(b), ci(c), wtl(c), f"{100 * c.mean_pp / a.mean_pp:.0f}\\%"])
    table("tab4_mixed_split", "tab:mixedsplit",
          "Разложение выигрыша DP-FGD",
          "lllll",
          "Домен & FGD-посл. \\(-\\) BestFit & DP-FGD \\(-\\) FGD-посл. & Л/Р/Х & Доля совместности",
          rows, "Разницы \\(U_{100}\\) в п.п., в скобках 95\\% ДИ. Л/Р/Х дано для DP-FGD "
                "против последовательного FGD.")

    rows = []
    for lab, nm in [("FirstFit - aBF", "FirstFit"), ("BestFit-GPU - aBF", "BestFit по GPU"),
                    ("DotProd - aBF", "DotProd"), ("Packing - aBF", "Packing"),
                    ("Random - aBF", "Random")]:
        r = get(lab, "ALL")
        rows.append([nm, ci(r), wtl(r)])
    table("tab4_mixed_heur", "tab:heur",
          "Эвристики против BestFit",
          "lll", "Политика & \\(-\\) BestFit, п.п. [95\\% ДИ] & Л/Р/Х", rows,
          "Разница \\(U_{100}\\) с BestFit по всем 776 блокам. Л/Р/Х: число блоков, "
          "где эвристика лучше, равна и хуже BestFit.")

    f1 = pd.read_csv(ROOT / "results" / "aggregates" / "f1" / "f1_contrasts.csv")

    def fget(label):
        return f1[(f1.contrast == label) & (f1.scope == "ALL")].iloc[0]

    rows = []
    for v, nm in [("full", "полная нагрузка"),
                  ("splitfracgang", "расщеплены дробные групповые"),
                  ("splitwholegang", "расщеплены целые групповые"),
                  ("splitall", "расщеплены все групповые")]:
        g = fget(f"G({v})")
        gap = "0 на каждом блоке" if g.max_abs_pp == 0 else ci(g)
        rows.append([nm, ci(fget(f"S({v})")), gap])
    table("tab4_f1", "tab:f1",
          "Расщепление групповых задач",
          "lll",
          "Вариант нагрузки & FGD-посл. \\(-\\) BestFit & DP-FGD \\(-\\) FGD-посл.",
          rows, "Разницы \\(U_{100}\\) в п.п. по всем 776 блокам, в скобках 95\\% ДИ. Блоки, спрос "
                "и порядок задач во всех вариантах одни и те же.")

    blocks = pd.read_csv(MIX / "mixed_blocks.csv")
    rows = []
    for s in SCOPES:
        b = blocks if s == "ALL" else blocks[blocks.domain == s]
        dem = b.demand_gpus.sum()
        if s == "ALL":
            rows.append("MID")
        rows.append([NAME[s], str(len(b)),
                     f"{b['items'].mean():.0f}", f"{b.gangs.mean():.0f}",
                     f"{(b.mean_gang_size * b.gangs).sum() / b.gangs.sum():.1f}",
                     f"{100 * (b.gang_demand_share * b.demand_gpus).sum() / dem:.1f}",
                     f"{100 * (b.fractional_gang_demand_share * b.demand_gpus).sum() / dem:.1f}",
                     f"{100 * (b.fractional_singleton_demand_share * b.demand_gpus).sum() / dem:.1f}"])
    table("tab3_blocks", "tab:blocks",
          "Блоки нагрузки по доменам",
          "lrrrrrrr",
          "Домен & Блоков & Элем. & Групп. & Размер & \\multicolumn{3}{c}{Доля спроса, \\%} \\\\\n"
          " & & в блоке & в блоке & групп. & групп. & дробн. групп. & дробн. одиночн.",
          rows, "Блоки набора данных PAI 2020 со всеми типами задач, средние по блокам. Доли считаются от спроса блоков на GPU. Элемент: "
                "групповая (групп.) или одиночная задача. Размер групповой задачи "
                "равен числу её подзадач.")

    rows = []
    for s in SCOPES:
        if s == "ALL":
            rows.append("MID")
        rows.append([NAME[s]] + [ci(get(f"{p}: {m} - oracle", s))
                                 for m in ("prev", "uniform") for p in ("DP", "seq")])
    table("tab4_m_cost", "tab:mcost",
          "Цена неточного M",
          "lllll",
          "Домен & \\multicolumn{2}{c}{M прошлого блока} & \\multicolumn{2}{c}{Все классы поровну} \\\\\n"
          " & DP-FGD & FGD-посл. & DP-FGD & FGD-посл.",
          rows, "Разница \\(U_{100}\\) с данным M и с точным M, п.п. Значение больше нуля "
                "означает, что с данным M результат хуже. В скобках 95\\% ДИ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
