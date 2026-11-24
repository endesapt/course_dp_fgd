#!/usr/bin/env python3
"""Tiny end-to-end smoke run: one gang stream, one policy, one seed."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stream = ROOT / "data" / "gang" / "synth_gangs_seed0_f00_mix_load1_frac28.jsonl"
    if not stream.exists():
        raise SystemExit(f"missing smoke stream: {stream}")
    smoke_dir = ROOT / "results" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    tiny = smoke_dir / "stream_smoke.jsonl"
    lines = stream.read_text(encoding="utf-8").splitlines()
    tiny.write_text("\n".join(lines[:201]) + "\n", encoding="utf-8")
    out = smoke_dir / "gang_dp"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "calibrate_gang.py"),
        "--policy", "dp",
        "--streams", str(tiny),
        "--seeds", "1",
        "--out", str(out),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    produced = list(out.glob("*.json")) + list(out.glob("**/*.json"))
    if not produced:
        raise SystemExit(f"smoke produced no artifacts under {out}")
    print(f"smoke ok: {len(produced)} artifact(s) under {out}")


if __name__ == "__main__":
    main()
