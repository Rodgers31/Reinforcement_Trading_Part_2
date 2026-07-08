"""Task 27 Track A — final XAUUSD lever family (enh/12 §1-§3; pinned pre-execution).

Cells (folds 11-25, 10y arm, V1 grammar):
  BASE : sign(score), |score| >= train-q80              (pre-flight vs committed V1_10y)
  L1   : longs q80; SHORTS require |score| >= train-q90 (existing Task-22 constant)
  L2   : long-only (longs q80; shorts never)
  BETA : always-long, every bar (mandatory control)     + unlevered gold B&H context row

Success (all four): full ratified gate; stitched return >= +63.1%; E4 return > 0;
beats-beta (return AND metric > BETA's). TERMINAL RULE: neither L1 nor L2 passes ->
XAUUSD-alone supervised line RESTS at benchmark. Deterministic; zero new constants.

Run: .venv/bin/python enhancements/12_levers/run_levers.py
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
sb.OUT = OUT
VIABILITY = 63.1
Q_LONG, Q_SHORT = 0.8, 0.9      # existing pinned constants (Task-22)
K = sb.K


def simulate_directional(te: pd.DataFrame, dir_arr: np.ndarray):
    """V1 grammar with an explicit per-bar entry direction (-1/0/+1)."""
    c = te["Close"].to_numpy(float)
    atr = te["atr"].to_numpy(float)
    times = te.index
    n = len(te)
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
        if pos is not None and i - pos["i"] >= K:
            close_trade(c[i], i, "time")
        if pos is None and dir_arr[i] != 0:
            d = int(dir_arr[i])
            atr_e = max(atr[i], 1e-12)
            entry = c[i] + d * HALF * atr_e
            risk = max(equity * sb.RISK_FRAC, 1e-8)
            pos = {"dir": d, "entry": entry, "mid": c[i], "units": risk / atr_e,
                   "risk": risk, "atr_e": atr_e, "i": i, "t": times[i]}
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
    nn = len(flb)
    fwd4 = np.full(nn, np.nan)
    fwd4[: nn - K] = (close[K:] - close[:-K]) / atr_all[: nn - K]
    X_all = flb[feature_cols].to_numpy(float)

    cells = {n: ([], [], []) for n in ("BASE", "L1", "L2", "BETA")}
    bh_first = bh_last = None
    bh_e4_first = None
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
        ptr = np.abs(model.predict(sc.transform(X_all[tr_sl][m])))
        thr80, thr90 = float(np.quantile(ptr, Q_LONG)), float(np.quantile(ptr, Q_SHORT))
        s = model.predict(sc.transform(X_all[te_sl]))

        dirs = {
            "BASE": np.sign(s) * (np.abs(s) >= thr80),
            "L1": np.where((s > 0) & (s >= thr80), 1,
                           np.where((s < 0) & (-s >= thr90), -1, 0)),
            "L2": np.where((s > 0) & (s >= thr80), 1, 0),
            "BETA": np.ones(len(s)),
        }
        for name, da in dirs.items():
            eq, tr_df = simulate_directional(te, da.astype(int))
            eqs, trs, summ = cells[name]
            eqs.append(eq)
            trs.append(tr_df)
            summ.append(sb.fold_row(i, eq, tr_df))
        tc = te["Close"]
        if bh_first is None:
            bh_first = float(tc.iloc[0])
        if i == 19:
            bh_e4_first = float(tc.iloc[0])
        bh_last = float(tc.iloc[-1])
        print(f"fold {i:>2} done ({time.time() - t0:,.0f}s)", flush=True)

    ref = pd.read_csv(REPO / "enhancements" / "09_probe" / "summary_V1_10y.csv")
    got = pd.DataFrame(cells["BASE"][2])
    for col in ["test_return_pct", "test_profit_factor", "fold_metric", "test_n_trades"]:
        diff = np.abs(got[col].to_numpy() - ref[col].to_numpy()).max()
        assert diff < 1e-9, f"pre-flight FAIL on {col}: {diff}"
    print("[OK] PRE-FLIGHT: BASE reproduces committed V1_10y per-fold rows exactly")

    report = {"pinned": {"q_long": Q_LONG, "q_short": Q_SHORT, "viability": VIABILITY},
              "gold_bh_context_pct": {
                  "stitched_window": round((bh_last / bh_first - 1) * 100, 1),
                  "E4_window": round((bh_last / bh_e4_first - 1) * 100, 1)}}
    res_all = {}
    for name in ("BASE", "L1", "L2", "BETA"):
        eqs, trs, summ = cells[name]
        res = sb.evaluate_variant(f"{name}_lever", eqs, trs, summ, sb.W10_FOLDS)
        res_all[name] = res
    beta = res_all["BETA"]
    for name in ("BASE", "L1", "L2", "BETA"):
        res = res_all[name]
        res["viability_pass"] = bool(res["stitched_return_pct"] >= VIABILITY)
        res["beats_beta"] = bool(
            res["stitched_return_pct"] > beta["stitched_return_pct"]
            and res["metric_stitched"] > beta["metric_stitched"]) if name != "BETA" else None
        res["FULL_PASS"] = bool(res["gate_passed"] and res["viability_pass"]
                                and (res["E4_return_pct"] or 0) > 0
                                and bool(res["beats_beta"])) if name != "BETA" else None
        report[name] = res
        bb = "n/a" if res["beats_beta"] is None else ("PASS" if res["beats_beta"] else "FAIL")
        fp = "n/a" if res["FULL_PASS"] is None else ("PASS" if res["FULL_PASS"] else "FAIL")
        print(f"\n=== {name}: metric {res['metric_stitched']:+.4f} | ret "
              f"{res['stitched_return_pct']:+.1f}% (viab {'PASS' if res['viability_pass'] else 'FAIL'}) "
              f"| PF {res['stitched_pf']:.3f} | gate {'PASS' if res['gate_passed'] else 'FAIL'} | "
              f"E4 {res['E4_return_pct']:+.1f}% | beats-beta {bb} | FULL {fp} | trades {res['n_trades']}")
        for line in res["gate_detail"]:
            print(f"    {line}")
        badf = pd.DataFrame(cells[name][2])
        badf = badf[badf.fold.isin([15, 16, 18, 24])]
        print("    f15/f16/f18/f24:",
              {int(r.fold): (round(r.test_return_pct, 1), round(r.test_profit_factor, 3))
               for r in badf.itertuples()})
    print(f"\ngold B&H context: stitched-window {report['gold_bh_context_pct']['stitched_window']}% "
          f"| E4-window {report['gold_bh_context_pct']['E4_window']}%")
    (OUT / "levers_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"runtime {time.time() - t0:,.0f}s; levers_report.json written")


if __name__ == "__main__":
    main()
