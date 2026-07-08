"""enh/11 §4 — E3 failure diagnosis (A2: runs FIRST; analysis-only).

Pre-registered questions Q1-Q4 over committed artifacts only. No parameter of
H1/H2/H3 may change in response to anything found here (no-parameter-change
rule, enh/11 §4). Output: printed tables + e3_diagnosis.json.

Run: .venv/bin/python enhancements/11_hardening/e3_diagnosis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P9 = REPO / "enhancements" / "09_probe"
P8 = REPO / "enhancements" / "08_probe"

BAD = [15, 16, 18]          # E3 losers in V1-10y
CONTEXT = [13, 19, 25]      # clearing folds for contrast
TOP_FEATS = ["dow_cos", "tod_cos", "lower_wick_ratio", "session_london",
             "atr_fast_slow", "close_ema200_atr"]


def main() -> None:
    tr = pd.read_csv(P9 / "trades_V1_10y.csv.gz")
    tr["entry_time"] = pd.to_datetime(tr.entry_time)
    win = pd.read_csv(P8 / "fold_windows.csv")
    win["test_start"] = pd.to_datetime(win.test_start)
    win["test_end"] = pd.to_datetime(win.test_end)
    uni = pd.read_csv(P8 / "results_univariate.csv")

    # map trades -> fold
    tr["fold"] = 0
    for _, w in win[win.fold >= 11].iterrows():
        m = (tr.entry_time >= w.test_start) & (tr.entry_time <= w.test_end)
        tr.loc[m, "fold"] = w.fold
    assert (tr.fold >= 11).all(), "unmapped trades"

    report = {}

    print("=" * 90)
    print("Q1 — LOSS CONCENTRATION (share of gross losses carried by the 10 worst trades)")
    q1 = {}
    for f in BAD + CONTEXT:
        d = tr[tr.fold == f]
        neg = d[d.pnl < 0].pnl
        worst10 = neg.nsmallest(10).sum()
        q1[f] = {"net_pnl": round(float(d.pnl.sum()), 1),
                 "gross_losses": round(float(neg.sum()), 1),
                 "worst10_sum": round(float(worst10), 1),
                 "worst10_share_of_losses_pct": round(float(worst10 / neg.sum() * 100), 1),
                 "n_trades": int(len(d))}
        tag = "BAD" if f in BAD else "ctx"
        print(f"  f{f} [{tag}]: net {q1[f]['net_pnl']:>8} | worst10 {q1[f]['worst10_share_of_losses_pct']:>5}% "
              f"of gross losses ({q1[f]['worst10_sum']} of {q1[f]['gross_losses']})")
    report["Q1_concentration"] = q1

    print("\nQ2 — DIRECTIONAL ASYMMETRY (E3 folds: net pnl + mean r by direction)")
    q2 = {}
    for f in [13, 14, 15, 16, 17, 18]:
        d = tr[tr.fold == f]
        row = {}
        for dirn, name in [(1, "long"), (-1, "short")]:
            dd = d[d.direction == dirn]
            row[name] = {"n": int(len(dd)), "net_pnl": round(float(dd.pnl.sum()), 1),
                         "mean_r": round(float(dd.r_mult.mean()), 4)}
        q2[f] = row
        print(f"  f{f}: long n={row['long']['n']} net {row['long']['net_pnl']:>8} r {row['long']['mean_r']:+.3f} | "
              f"short n={row['short']['n']} net {row['short']['net_pnl']:>8} r {row['short']['mean_r']:+.3f}")
    report["Q2_direction"] = q2

    print("\nQ3 — SIGNAL PRESENCE (univariate era-median OOS IC, Task-22 data)")
    u = uni[uni.feature.isin(TOP_FEATS)]
    piv = u.pivot_table(index="feature", columns="era", values="ic_oos", aggfunc="median")
    print(piv.round(4).to_string())
    report["Q3_univariate_era_ic"] = {f: {str(e): round(float(piv.loc[f, e]), 4)
                                          for e in piv.columns} for f in piv.index}
    e3_alive = (piv[3].abs() >= 0.5 * piv[[1, 2]].abs().mean(axis=1)).sum()
    report["Q3_features_alive_in_E3"] = int(e3_alive)
    print(f"  features with |E3 IC| >= half their E1/E2 mean: {e3_alive}/{len(piv)}")

    print("\nQ4 — VOL REGIME (per-fold mean entry_atr/entry_price, percentile among f11-25)")
    volp = tr.groupby("fold").apply(lambda d: (d.entry_atr / d.entry_price).mean() * 100)
    ranks = volp.rank(pct=True) * 100
    q4 = {int(f): {"atr_pct_of_price": round(float(volp[f]), 4),
                   "percentile": round(float(ranks[f]), 0)} for f in volp.index}
    report["Q4_vol_regime"] = q4
    for f in BAD + [24]:
        print(f"  f{f}: ATR {volp[f]:.3f}% of price -> {ranks[f]:.0f}th pct "
              f"{'(BAD fold)' if f in BAD or f == 24 else ''}")

    # Pre-registered mechanism mapping (enh/11 A2): which H-cells are
    # mechanism-confirmed by the answers.
    tail_driven = all(q1[f]["worst10_share_of_losses_pct"] >= 25.0 for f in BAD)
    bad_high_vol = np.median([q4[f]["percentile"] for f in BAD + [24]]) >= 60.0
    report["mechanism_reading"] = {
        "H1_stop_mechanism_confirmed_if_tail_driven": bool(tail_driven),
        "H2_sizing_mechanism_confirmed_if_bad_folds_high_vol": bool(bad_high_vol),
        "E3_signal_alive_features": int(e3_alive),
    }
    print("\nMECHANISM READING (pre-registered):")
    print(f"  tail-driven losses in all bad folds (>=25% worst10 share): {tail_driven} -> H1 "
          f"{'mechanism-confirmed' if tail_driven else 'mechanism-SUSPICIOUS'}")
    print(f"  bad folds sit high-vol (median pct >= 60): {bad_high_vol} -> H2 "
          f"{'mechanism-confirmed' if bad_high_vol else 'mechanism-SUSPICIOUS'}")

    (OUT / "e3_diagnosis.json").write_text(json.dumps(report, indent=2))
    print("\ne3_diagnosis.json written")


if __name__ == "__main__":
    main()
