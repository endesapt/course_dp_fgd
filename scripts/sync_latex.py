#!/usr/bin/env python3
"""Copy generated figures/tables into the sibling LaTeX project."""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATEX = ROOT.parent / "latex_coursework"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", default=str(DEFAULT_LATEX))
    args = ap.parse_args()
    latex = Path(args.latex)
    if not latex.exists():
        raise SystemExit(f"latex dir not found: {latex}")
    img = latex / "images"
    tab = latex / "tables"
    img.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in sorted((ROOT / "results" / "figures").glob("fig4_*")):
        shutil.copyfile(p, img / p.name)
        n += 1
    for p in sorted((ROOT / "results" / "tables").glob("*.tex")):
        shutil.copyfile(p, tab / p.name)
        n += 1
    print(f"synced {n} file(s) -> {latex}")


if __name__ == "__main__":
    main()
