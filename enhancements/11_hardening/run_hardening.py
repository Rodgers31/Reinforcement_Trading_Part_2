"""enh/11 §2 hardening cells H1/H2/H3 (Task 26 Track 2; runs AFTER e3_diagnosis per A2).

Constants frozen in enh/11 §2 (declared before any run):
  H1: protective SL 2.0xATR_entry (intrabar H1-touch, gap-through at open, NO TP),
      k=4 time exit unchanged, sizing unchanged.
  H2: per-entry risk scale max(0.25, min(1, median_train(atr_close)/atr_close[t])), no stop.
  H3: both.
Base = Task-23 V1-10y (fwd4-ridge top-quintile, sign, 4-bar exit), folds 11-25.

PRE-FLIGHT: with stop off + scale 1 this runner must reproduce the committed
summary_V1_10y.csv per-fold rows exactly — proving faithfulness before any H-cell.

Success (enh/11 §3, amended): FULL ratified gate + E4 return > 0 + stitched
return >= +63.1% (A1). Passing cell -> enh/12 proposal (A3). None pass -> line stops.

Run: .venv/bin/python enhancements/11_hardening/run_hardening.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "enhancements" / "09_probe"))

import run_supervised_baseline as sb  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import CFG  # noqa: E402
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402

OUT = Path(__file__).resolve().parent
sb.OUT = OUT                          # cell artifacts land in 11_hardening/
STOP_MULT = 2.0                       # H1/H3 protective stop (xATR_entry)
SCALE_FLOOR = 0.25                    # H2/H3 sizing floor
VIABILITY_RETURN_PCT = 63.1           # A1 economic-viability floor
K = sb.K                              # 4-bar time exit (frozen)


def simulate_hardened(te, scores, thr, stop_mult=None, scale=None):
    """V1 time-exit simulator + optional protective stop + optional size scale.

    Mirrors sb.simulate's V1 semantics exactly when stop_mult is None and
    scale is None (asserted by the pre-flight)."""
    o = te["Open"].to_numpy(float)
    h = te["High"].to_numpy(float)
    l = te["Low"].to_numpy(float)
    c = te["Close"].to_numpy(float)
    atr = te["atr"].to_numpy(float)
    times = te.index
    n = len(te)
    selected = np.abs(scores) >= thr
    HALF = sb.HALF_COST_ATR

    equity = sb.INIT_EQ
    pos = None
    eq_r = np.empty(n)
    eq_m = np.empty(n)
    trades = []

    def close_trade(raw, i, reason):
        nonlocal equity, pos
        exit_price = raw - pos["dir"] * HALF * pos["atr_e"]
        pnl = (exit_price - pos["entry"]) * pos["units"] * pos["dir"] - sb.COMMISSION
        gross = (raw - pos["mid"]) * pos["units"] * pos["dir"]
        equity += pnl
        trades.append({
            "entry_time": pos["t"], "exit_time": times[i], "direction": pos["dir"],
            "entry_price": pos["entry"], "exit_price": exit_price,
            "entry_atr": pos["atr_e"], "units": pos["units"], "pnl": pnl,
            "gross_pnl_mid": gross, "cost_paid": gross - pnl,
            "r_mult": pnl / pos["risk"], "bars_in_trade": i - pos["i"],
            "exit_reason": reason,
        })
        pos = None

    for i in range(n):
        if pos is not None:
            d = pos["dir"]
            if stop_mult is not None and i > pos["i"]:
                sl = pos["sl"]
                raw = reason = None
                if (d == 1 and o[i] <= sl) or (d == -1 and o[i] >= sl):
                    raw, reason = o[i], "SL_gap"
                elif (d == 1 and l[i] <= sl) or (d == -1 and h[i] >= sl):
                    raw, reason = sl, "SL"
                if raw is not None:
                    close_trade(raw, i, reason)
            if pos is not None and i - pos["i"] >= K:
                close_trade(c[i], i, "time")
        if pos is None and selected[i]:
            d = 1 if scores[i] > 0 else -1
            atr_e = max(atr[i], 1e-12)
            entry = c[i] + d * HALF * atr_e
            risk = max(equity * sb.RISK_FRAC * (scale[i] if scale is not None else 1.0),
                       1e-8)
            units = risk / (1.0 * atr_e)          # risk unit stays 1xATR (enh/11 §2)
            pos = {"dir": d, "entry": entry, "mid": c[i], "units": units,
                   "risk": risk, "atr_e": atr_e, "i": i, "t": times[i],
                   "sl": entry - d * STOP_MULT * atr_e}
        if pos is not None and i == n - 1:
            close_trade(c[i], i, "eow")
        eq_r[i] = equity
        eq_m[i] = equity + (0.0 if pos is None else
                            (c[i] - pos["entry"]) * pos["units"] * pos["dir"])

    return (pd.DataFrame({"equity": eq_r, "equity_mtm": eq_m}, index=times),
            pd.DataFrame(trades))


def main() -> None:
    t0 = time.time()
    print("Loading data (anchor-identical path)…")
    _, feat, feature_cols = _load_decision_features()
    folds = make_sliding_folds(feat, train_years=CFG.sliding_train_years,
                               val_months=CFG.sliding_val_months,
                               test_months=CFG.sliding_test_months,
                               step_months=CFG.sliding_step_months,
                               embargo_bars=CFG.split_embargo_bars,
                               lockbox_start=CFG.lockbox_start_date)
    assert len(folds) == 25 and CFG.sliding_train_years == 5.0
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index
    bounds = sb.derive_boundaries(idx)
    emb = CFG.split_embargo_bars

    close = flb["Close"].to_numpy(float)
    atr_all = flb["atr"].to_numpy(float)
    atr_close_all = flb["atr_close"].to_numpy(float)
    nn = len(flb)
    fwd4 = np.full(nn, np.nan)
    fwd4[: nn - K] = (close[K:] - close[:-K]) / atr_all[: nn - K]
    X_all = flb[feature_cols].to_numpy(float)

    cells = {"BASE": ([], [], []), "H1": ([], [], []),
             "H2": ([], [], []), "H3": ([], [], [])}
    for i, (t_0, a, b, c_, d) in enumerate(bounds, start=1):
        if i not in sb.W10_FOLDS:
            continue
        te = folds[i - 1][2]
        te_sl = slice(min(c_ + emb, d), d)
        aw = int(idx.searchsorted(t_0 - pd.DateOffset(months=60), side="left"))
        tr_sl = slice(aw, max(b - emb, aw))
        ytr = fwd4[tr_sl]
        m = ~np.isnan(ytr)
        sc = StandardScaler().fit(X_all[tr_sl][m])
        model = Ridge(alpha=1.0).fit(sc.transform(X_all[tr_sl][m]), ytr[m])
        thr = float(np.quantile(np.abs(model.predict(sc.transform(X_all[tr_sl][m]))), sb.Q))
        scores = model.predict(sc.transform(X_all[te_sl]))
        # H2/H3 scale: train-median atr_close vs test-bar atr_close (frozen formula)
        m_train = float(np.median(atr_close_all[tr_sl]))
        scale = np.maximum(SCALE_FLOOR,
                           np.minimum(1.0, m_train / atr_close_all[te_sl]))
        for name, kw in [("BASE", {}), ("H1", dict(stop_mult=STOP_MULT)),
                         ("H2", dict(scale=scale)),
                         ("H3", dict(stop_mult=STOP_MULT, scale=scale))]:
            eq, tr_df = simulate_hardened(te, scores, thr, **kw)
            eqs, trs, summ = cells[name]
            eqs.append(eq)
            trs.append(tr_df)
            summ.append(sb.fold_row(i, eq, tr_df))
        print(f"fold {i:>2} done ({time.time() - t0:,.0f}s)", flush=True)

    # PRE-FLIGHT: BASE must equal the committed V1-10y rows.
    ref = pd.read_csv(REPO / "enhancements" / "09_probe" / "summary_V1_10y.csv")
    got = pd.DataFrame(cells["BASE"][2])
    for col in ["test_return_pct", "test_profit_factor", "fold_metric", "test_n_trades"]:
        diff = np.abs(got[col].to_numpy() - ref[col].to_numpy()).max()
        assert diff < 1e-9, f"pre-flight FAIL on {col}: max diff {diff}"
    print("[OK] PRE-FLIGHT: BASE reproduces committed V1_10y per-fold rows exactly")

    report = {"pinned": {"stop_mult": STOP_MULT, "scale_floor": SCALE_FLOOR,
                         "viability_return_pct": VIABILITY_RETURN_PCT, "k": K}}
    for name in ["BASE", "H1", "H2", "H3"]:
        eqs, trs, summ = cells[name]
        res = sb.evaluate_variant(f"{name}_10y", eqs, trs, summ, sb.W10_FOLDS)
        res["viability_floor_pass"] = bool(res["stitched_return_pct"] >= VIABILITY_RETURN_PCT)
        res["FULL_PASS"] = bool(res["gate_passed"]
                                and (res["E4_return_pct"] or 0) > 0
                                and res["viability_floor_pass"])
        report[name] = res
        print(f"\n=== {name}: metric {res['metric_stitched']:+.4f} | ret "
              f"{res['stitched_return_pct']:+.1f}% (floor {VIABILITY_RETURN_PCT}: "
              f"{'PASS' if res['viability_floor_pass'] else 'FAIL'}) | maxDD "
              f"{res['stitched_max_dd_mtm_pct']:.1f}% | PF {res['stitched_pf']:.3f} | gate "
              f"{'PASS' if res['gate_passed'] else 'FAIL'} | E4 {res['E4_return_pct']:+.1f}% | "
              f"FULL {'PASS' if res['FULL_PASS'] else 'FAIL'}")
        for line in res["gate_detail"]:
            print(f"    {line}")
        bad = pd.DataFrame(summ)
        bad = bad[bad.fold.isin([15, 16, 18, 24])]
        print("    f15/f16/f18/f24:",
              {int(r.fold): (round(r.test_return_pct, 1), round(r.test_profit_factor, 3))
               for r in bad.itertuples()})
    (OUT / "hardening_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nruntime {time.time() - t0:,.0f}s; hardening_report.json written")


if __name__ == "__main__":
    main()
