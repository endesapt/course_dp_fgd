#!/usr/bin/env python3
"""Chapter-4 statistics for the mixed PAI blocks + whole-only reference, and
the matching rows of results/claims.csv (ids N01.., regenerated each run).

Reads only committed aggregates:
  results/aggregates/mixed/mixed_cells.csv, mixed_blocks.csv
  results/aggregates/pai_l1_v2/pai_l1_v2_all_cells.csv  (whole-only, stage=core)
Writes:
  results/aggregates/mixed/mixed_domain_means.csv
  results/aggregates/mixed/mixed_contrasts.csv
  results/claims.csv  (rows N* replaced, other rows untouched)
If results/aggregates/f1/f1_cells.csv exists (scripts/collect_f1_cells.py),
also writes results/aggregates/f1/f1_contrasts.csv and appends the F1 claims
(composition of the DP - seq gain, pre-registered H1-H3, Holm within scope).

Statistics: paired by block (matched blocks only), cluster bootstrap over 10
consecutive blocks within a domain (4000 resamples, seed 0), two-sided
bootstrap p, Holm within each family at scope ALL.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "aggregates"
MIX = AGG / "mixed"
CLAIMS = ROOT / "results" / "claims.csv"
SIG = ["MISC", "V100", "V100M32"]
SCOPES = ["ALL"] + SIG
N_BOOT, SEED = 4000, 0

# family, label, minuend arm, subtrahend arm  (arms of mixed_cells)
CONTRASTS = [
    ("policy", "DP - aBF", "dp_stream", "abf_stream"),
    ("policy", "seq - aBF", "seq_stream", "abf_stream"),
    ("policy", "DP - seq", "dp_stream", "seq_stream"),
    ("heuristic", "FirstFit - aBF", "aff_stream", "abf_stream"),
    ("heuristic", "BestFit-GPU - aBF", "abfgpu_stream", "abf_stream"),
    ("heuristic", "DotProd - aBF", "adot_stream", "abf_stream"),
    ("heuristic", "Packing - aBF", "apack_stream", "abf_stream"),
    ("heuristic", "Random - aBF", "arand_stream", "abf_stream"),
    ("m_cost", "DP: prev - oracle", "dp_prev", "dp_stream"),
    ("m_cost", "seq: prev - oracle", "seq_prev", "seq_stream"),
    ("m_cost", "DP: uniform - oracle", "dp_uniform", "dp_stream"),
    ("m_cost", "seq: uniform - oracle", "seq_uniform", "seq_stream"),
]
WHOLE = [
    ("whole", "whole DP - aBF", "dp", "atomic_bestfit"),
    ("whole", "whole seq - aBF", "seq_fgd", "atomic_bestfit"),
    ("whole", "whole DP - seq", "dp", "seq_fgd"),
]


def boot(d: pd.Series):
    rng = np.random.default_rng(SEED)
    keys = [f"{dom}:{b // 10}" for dom, b in d.index]
    g = pd.DataFrame({"d": d.values, "c": keys}).groupby("c")["d"].agg(["sum", "count"])
    s, n = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(p)


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def contrast_rows(wide: pd.DataFrame, specs, source: str) -> list[dict]:
    rows = []
    for fam, label, a, b in specs:
        if a not in wide or b not in wide:
            continue
        d_all = (wide[a] - wide[b]).dropna()
        for scope in SCOPES:
            d = d_all if scope == "ALL" else d_all[d_all.index.get_level_values(0) == scope]
            if len(d) < 5:
                continue
            m, lo, hi, p = boot(d)
            base = wide[b].reindex(d.index).mean()
            rows.append({
                "family": fam, "contrast": label, "scope": scope, "n": len(d),
                "mean_pp": m, "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                "rel_pct": 100 * m / base if base else np.nan,
                "wins": int((d < -1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()),
                "losses": int((d > 1e-9).sum()), "source": source,
            })
    return rows


M_ARMS = {"dp": ("dp_stream", "dp_prev", "dp_uniform"),
          "seq_fgd": ("seq_stream", "seq_prev", "seq_uniform")}
M_KIND = {"stream": "oracle", "prev": "prev", "uniform": "uniform"}


def m_levels(wide: pd.DataFrame) -> pd.DataFrame:
    """Mean unalloc@100 per (policy, M kind) on blocks where all six arms exist."""
    arms = [a for t in M_ARMS.values() for a in t]
    sub = wide[arms].dropna()
    rows = []
    for scope in SCOPES:
        s = sub if scope == "ALL" else sub[sub.index.get_level_values(0) == scope]
        for pol, t in M_ARMS.items():
            for arm in t:
                rows.append({"scope": scope, "policy": pol,
                             "m": M_KIND[arm.split("_", 1)[1]], "n": len(s),
                             "unalloc100": s[arm].mean()})
    return pd.DataFrame(rows)


F1 = AGG / "f1"
F1_VARIANTS = ["full", "splitfracgang", "splitwholegang", "splitall", "splitfracgang_fgdbg"]


def f1_contrasts(cells: pd.DataFrame) -> pd.DataFrame | None:
    """F1: G(v) = DP - seq per variant and the pre-registered differences
    H1 = G(full) - G(splitfracgang), H2 = G(full) - G(splitwholegang),
    H3 = G(splitfracgang_fgdbg) - G(splitfracgang); `full` = oracle-M mixed arms."""
    path = F1 / "f1_cells.csv"
    if not path.exists():
        return None
    f1 = pd.read_csv(path)
    full = cells[cells.arm.isin(["dp_stream", "seq_stream", "abf_stream"])].assign(variant="full")
    df = pd.concat([f1, full[f1.columns]], ignore_index=True)
    w = df.pivot_table(index=["domain", "block"], columns=["variant", "policy"], values="unalloc100")
    G = {v: w[(v, "dp")] - w[(v, "seq_fgd")] for v in F1_VARIANTS}
    S = {v: w[(v, "seq_fgd")] - w[(v, "atomic_bestfit")] for v in F1_VARIANTS}
    specs = [("gap", f"G({v})", G[v], None) for v in F1_VARIANTS]
    specs += [("gap", f"S({v})", S[v], None) for v in F1_VARIANTS]
    specs += [("H", "H1 G(full) - G(splitfracgang)", G["full"] - G["splitfracgang"], G["full"]),
              ("H", "H2 G(full) - G(splitwholegang)", G["full"] - G["splitwholegang"], G["full"]),
              ("H", "H3 G(splitfracgang_fgdbg) - G(splitfracgang)",
               G["splitfracgang_fgdbg"] - G["splitfracgang"], None),
              ("desc", "S(full) - S(splitall)", S["full"] - S["splitall"], None)]
    rows = []
    for fam, label, d_all, ref in specs:
        d_all = d_all.dropna()
        for scope in SCOPES:
            d = d_all if scope == "ALL" else d_all[d_all.index.get_level_values(0) == scope]
            m, lo, hi, p = boot(d)
            share = m / ref.reindex(d.index).mean() if ref is not None else np.nan
            rows.append({"family": fam, "contrast": label, "scope": scope, "n": len(d),
                         "mean_pp": m, "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                         "share_of_full_gap": share, "max_abs_pp": d.abs().max()})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for scope, g in res[res.family == "H"].groupby("scope"):
        res.loc[g.index, "p_holm"] = holm(g.p_boot.tolist())
    res.to_csv(F1 / "f1_contrasts.csv", index=False, float_format="%.4f")
    return res


def main() -> int:
    cells = pd.read_csv(MIX / "mixed_cells.csv")
    wide = cells.pivot_table(index=["domain", "block"], columns="arm", values="unalloc100")
    means = cells.groupby(["arm", "policy", "m", "domain"]).agg(
        n=("unalloc100", "count"), unalloc100=("unalloc100", "mean"),
        unalloc_end=("unalloc_end", "mean")).reset_index()
    means.to_csv(MIX / "mixed_domain_means.csv", index=False, float_format="%.4f")
    m_levels(wide).to_csv(MIX / "mixed_m_levels.csv", index=False, float_format="%.4f")

    whole = pd.read_csv(AGG / "pai_l1_v2" / "pai_l1_v2_all_cells.csv")
    whole = whole[(whole.stage == "core") & whole.domain.isin(SIG)]
    wwide = whole.pivot_table(index=["domain", "block"], columns="policy", values="unalloc100")

    rows = contrast_rows(wide, CONTRASTS, "mixed_cells.csv")
    rows += contrast_rows(wwide, WHOLE, "pai_l1_v2_all_cells.csv")
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for (fam, scope), g in res.groupby(["family", "scope"]):
        if scope == "ALL":
            res.loc[g.index, "p_holm"] = holm(g.p_boot.tolist())
    res.to_csv(MIX / "mixed_contrasts.csv", index=False, float_format="%.4f")
    print(res[res.scope == "ALL"][["contrast", "n", "mean_pp", "ci_lo", "ci_hi",
                                   "p_holm", "wins", "ties", "losses"]].round(3).to_string())

    write_claims(res, means, f1_contrasts(cells))
    return 0


def write_claims(res: pd.DataFrame, means: pd.DataFrame, f1: pd.DataFrame | None = None) -> None:
    src_c = "results/aggregates/mixed/mixed_contrasts.csv"
    src_m = "results/aggregates/mixed/mixed_domain_means.csv"
    src_b = "results/aggregates/mixed/mixed_blocks.csv"
    out = []

    def add(chapter, claim, value, unit, source, filt, used):
        out.append({"claim_id": f"N{len(out) + 1:02d}", "chapter": chapter, "claim": claim,
                    "value": value, "unit": unit, "source_file": source,
                    "filter": filt, "used_in": used})

    def get(label, scope="ALL"):
        return res[(res.contrast == label) & (res.scope == scope)].iloc[0]

    def ci(r):
        return f"{r.mean_pp:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"

    blocks = pd.read_csv(MIX / "mixed_blocks.csv")
    gang = (blocks.gang_demand_share * blocks.demand_gpus).sum()
    fgang = (blocks.fractional_gang_demand_share * blocks.demand_gpus).sum()
    add("3", "mixed blocks per domain",
        "/".join(str((blocks.domain == d).sum()) for d in SIG), "count", src_b,
        "MISC/V100/V100M32", "tab3_blocks")
    add("3", "fractional-gang share of block GPU demand",
        "/".join(f"{100 * (b.fractional_gang_demand_share * b.demand_gpus).sum() / b.demand_gpus.sum():.1f}"
                 for _, b in blocks.groupby("domain")), "%", src_b,
        "MISC/V100/V100M32 demand-weighted", "tab3_blocks")
    add("1", "fractional gangs share of gang GPU demand (blocks)",
        f"{100 * fgang / gang:.1f}", "%", src_b, "all 778 blocks", "text 1")

    for dom in SIG:
        v = {a: means[(means.arm == a) & (means.domain == dom)].unalloc100.iloc[0]
             for a in ("abf_stream", "seq_stream", "dp_stream")}
        add("4.1", f"unalloc@100 aBF -> seq -> DP, {dom}",
            f"{v['abf_stream']:.2f} -> {v['seq_stream']:.2f} -> {v['dp_stream']:.2f}",
            "%", src_m, f"{dom}, oracle M", "fig4_mixed_domains")

    for label, chap, fig in [("DP - aBF", "4.1", "fig4_mixed_domains"),
                             ("seq - aBF", "4.2", "fig4_mixed_split"),
                             ("DP - seq", "4.2", "fig4_mixed_split")]:
        r = get(label)
        add(chap, f"mixed {label} (ALL)", ci(r), "pp", src_c, "ALL, matched blocks, oracle M", fig)
        add(chap, f"mixed {label} W/T/L", f"{r.wins}/{r.ties}/{r.losses}", "count", src_c,
            "ALL", fig)
        add(chap, f"mixed {label} relative to subtrahend", f"{r.rel_pct:.0f}", "%", src_c,
            "ALL", fig)
        add(chap, f"mixed {label} by domain",
            "/".join(f"{get(label, d).mean_pp:+.2f}" for d in SIG), "pp", src_c,
            "MISC/V100/V100M32", fig)
    a, s = get("DP - aBF").mean_pp, get("seq - aBF").mean_pp
    add("4.2", "share of DP gain from FGD score (seq - aBF)/(DP - aBF)",
        f"{100 * s / a:.0f}", "%", src_c, "ALL", "fig4_mixed_split")
    add("4.2", "share of DP gain from joint placement (DP - seq)/(DP - aBF)",
        f"{100 * (a - s) / a:.0f}", "%", src_c, "ALL", "fig4_mixed_split")

    hs = res[(res.family == "heuristic") & (res.scope == "ALL")]
    det = hs[hs.contrast != "Random - aBF"]
    add("4.1", "max |heuristic - aBF| (FirstFit, BestFit-GPU, DotProd, Packing)",
        f"{det.mean_pp.abs().max():.3f}", "pp", src_c, "ALL", "fig4_mixed_domains")
    r = get("Random - aBF")
    add("4.1", "Random - aBF", ci(r), "pp", src_c, "ALL", "text 4.1")

    for label in ("whole DP - aBF", "whole seq - aBF", "whole DP - seq"):
        r = get(label)
        add("4.2", f"{label} (whole-GPU blocks)", ci(r), "pp", src_c,
            "signal domains, whole-only core", "not cited (former fig4_whole_vs_mixed)")
    for label in ("DP - aBF", "DP - seq"):
        ratio = get(label).mean_pp / get(f"whole {label}").mean_pp
        add("4.2", f"mixed / whole ratio of {label}", f"{ratio:.1f}", "x", src_c,
            "ALL (different block boundaries, unpaired)", "not cited (former fig4_whole_vs_mixed)")

    for label in ("DP: prev - oracle", "seq: prev - oracle",
                  "DP: uniform - oracle", "seq: uniform - oracle"):
        r = get(label)
        add("4.3", f"cost of M {label}", ci(r), "pp", src_c, "ALL, matched blocks",
            "fig4_m_cost")
        add("4.3", f"cost of M {label} by domain",
            "/".join(f"{get(label, d).mean_pp:+.2f}" for d in SIG), "pp", src_c,
            "MISC/V100/V100M32", "fig4_m_cost")

    lv = pd.read_csv(MIX / "mixed_m_levels.csv")
    lv = lv[lv.scope == "ALL"]
    for pol, name in (("dp", "DP"), ("seq_fgd", "seq")):
        v = {r.m: r.unalloc100 for r in lv[lv.policy == pol].itertuples()}
        add("4.3", f"unalloc@100 {name}: oracle / prev / uniform M",
            f"{v['oracle']:.2f} / {v['prev']:.2f} / {v['uniform']:.2f}", "%",
            "results/aggregates/mixed/mixed_m_levels.csv",
            f"ALL, blocks with all six M arms (n={int(lv.n.iloc[0])})", "fig4_m_cost")

    ts_path = MIX / "pai_trace_stats.csv"
    if ts_path.exists():
        ts = dict(pd.read_csv(ts_path).values)
        add("1", "gangs share of GPU demand (whole trace, 5 domains)",
            f"{ts['gang_share_of_gpu_demand_pct']:.1f}", "%",
            "results/aggregates/mixed/pai_trace_stats.csv", "mixed filter, 5 domains", "text 1")
        add("1", "fractional gangs share of gang GPU demand (whole trace, 5 domains)",
            f"{ts['fractional_gang_share_of_gang_demand_pct']:.1f}", "%",
            "results/aggregates/mixed/pai_trace_stats.csv", "mixed filter, 5 domains", "text 1")

    if f1 is not None:
        src_f = "results/aggregates/f1/f1_contrasts.csv"

        def fget(label, scope="ALL"):
            return f1[(f1.contrast == label) & (f1.scope == scope)].iloc[0]

        def fdom(label):
            return "/".join(f"{fget(label, d).mean_pp:+.2f}" for d in SIG)

        r = fget("G(splitall)")
        add("4.2", "F1 sanity: max |DP - seq| with all gangs split into singles",
            f"{max(fget('G(splitall)', d).max_abs_pp for d in SIG):.2f}", "pp", src_f,
            "MISC/V100/V100M32, every block", "text 4.2.1")
        for v, name in (("splitfracgang", "fractional gangs split"),
                        ("splitwholegang", "whole gangs split"),
                        ("splitfracgang_fgdbg", "fractional gangs split, FGD background")):
            add("4.2", f"F1 DP - seq, {name} (ALL)", ci(fget(f"G({v})")), "pp", src_f,
                "ALL, 778 mixed blocks, oracle M", "text 4.2.1")
            add("4.2", f"F1 DP - seq, {name} by domain", fdom(f"G({v})"), "pp", src_f,
                "MISC/V100/V100M32", "text 4.2.1")
        for h, name in (("H1 G(full) - G(splitfracgang)", "fractional gangs"),
                        ("H2 G(full) - G(splitwholegang)", "whole gangs")):
            add("4.2", f"F1 part of DP - seq due to {name} (ALL)", ci(fget(h)), "pp", src_f,
                "ALL, matched blocks", "text 4.2.1")
            add("4.2", f"F1 share of DP - seq due to {name}, ALL / MISC / V100 / V100M32",
                " / ".join(f"{100 * fget(h, s).share_of_full_gap:.0f}" for s in SCOPES), "%",
                src_f, "share of G(full) on the same blocks", "text 4.2.1")
        h3 = "H3 G(splitfracgang_fgdbg) - G(splitfracgang)"
        add("4.2", "F1 change of DP - seq with FGD background (ALL)", ci(fget(h3)), "pp", src_f,
            "ALL, matched blocks", "text 4.2.1")
        add("4.2", "F1 change of DP - seq with FGD background by domain", fdom(h3), "pp", src_f,
            "MISC/V100/V100M32", "text 4.2.1")
        add("4.2", "F1 seq - aBF: full minus all gangs split (ALL)",
            ci(fget("S(full) - S(splitall)")), "pp", src_f, "ALL, matched blocks", "text 4.2.1")

    # control domains P100/T4 (section 4.4): whole-GPU blocks, all policies give the same unalloc@100
    src_w = "results/aggregates/pai_l1_v2/pai_l1_v2_all_cells.csv"
    w = pd.read_csv(ROOT / src_w)
    w = w[(w.stage == "core") & (w.arm == "stream")]
    for dom in ("P100", "T4"):
        m = w[w.domain == dom].groupby("policy").unalloc100.mean()
        add("4.4", f"{dom} unalloc@100, all policies (control)", f"{m.mean():.2f}", "%", src_w,
            f"stage=core, {dom}, max-min over {len(m)} policies = {m.max() - m.min():.3f}", "text 4.4")

    old = pd.read_csv(CLAIMS, dtype=str)
    old = old[~old.claim_id.str.startswith("N")]
    pd.concat([old, pd.DataFrame(out)], ignore_index=True).to_csv(CLAIMS, index=False)
    print(f"claims.csv: {len(out)} N-rows written ({len(old)} other rows kept)")


if __name__ == "__main__":
    raise SystemExit(main())
