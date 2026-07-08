"""Task 27 Track B step 3 — the FROZEN V1 rule through the honest ruler on silver.

V1 exactly as pinned (fwd4-ridge, train-q80 |score|, sign direction, k=4 time
exit, 6mo-refit fold grid), 5y and 10y arms, silver's own measured cost bar,
the anchor's ruler functions (stitch, metric, gate). Reported with the standard
table + last-quartile ("E4-analog") subset; no pass/fail beyond the Track-27
replication verdict is pinned — these numbers are REPORTED.

Run: .venv/bin/python enhancements/12_silver/v1_silver.py
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

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import CFG  # noqa: E402
from data_loader import load_mt_ohlcv_csv, make_sliding_folds, resample_ohlcv  # noqa: E402
from eval_harness import metric_return_over_mtm_dd  # noqa: E402
from evaluate import full_report  # noqa: E402
from features import prepare_feature_frame  # noqa: E402
from run_baseline import _stitch  # noqa: E402
from train_ppo import _passes_consistency_gate  # noqa: E402

OUT = Path(__file__).resolve().parent
CC = json.loads((OUT / "silver_census_cost.json").read_text())
HALF = float(CC["spread_atr_frac"]) / 2.0 + float(CC["slippage_atr_frac"])
K, Q = 4, 0.8
INIT_EQ, RISK = CFG.initial_equity, CFG.risk_fraction


def derive_boundaries(idx):
    start, end = idx.min(), idx.max()
    tr_off = pd.DateOffset(months=int(round(CFG.sliding_train_years * 12)))
    va_off = pd.DateOffset(months=CFG.sliding_val_months)
    te_off = pd.DateOffset(months=CFG.sliding_test_months)
    st_off = pd.DateOffset(months=CFG.sliding_step_months)
    out, t0 = [], start
    while True:
        vs = t0 + tr_off
        ts = vs + va_off
        te = ts + te_off
        if te > end:
            break
        out.append((t0, int(idx.searchsorted(t0, "left")), int(idx.searchsorted(vs, "left")),
                    int(idx.searchsorted(ts, "left")), int(idx.searchsorted(te, "left"))))
        t0 = t0 + st_off
        if t0 + tr_off >= end:
            break
    return out


def simulate_v1(te: pd.DataFrame, dir_arr: np.ndarray):
    c = te["Close"].to_numpy(float)
    atr = te["atr"].to_numpy(float)
    times = te.index
    n = len(te)
    equity, pos = INIT_EQ, None
    eq_r, eq_m, trades = np.empty(n), np.empty(n), []

    def close_trade(raw, i, reason):
        nonlocal equity, pos
        exit_price = raw - pos["dir"] * HALF * pos["atr_e"]
        pnl = (exit_price - pos["entry"]) * pos["units"] * pos["dir"] - CFG.commission_per_trade
        equity += pnl
        trades.append({"pnl": pnl, "r_mult": pnl / pos["risk"],
                       "bars_in_trade": i - pos["i"], "exit_reason": reason,
                       "direction": pos["dir"]})
        pos = None

    for i in range(n):
        if pos is not None and i - pos["i"] >= K:
            close_trade(c[i], i, "time")
        if pos is None and dir_arr[i] != 0:
            d = int(dir_arr[i])
            atr_e = max(atr[i], 1e-12)
            risk = max(equity * RISK, 1e-8)
            pos = {"dir": d, "entry": c[i] + d * HALF * atr_e, "units": risk / atr_e,
                   "risk": risk, "atr_e": atr_e, "i": i}
        if pos is not None and i == n - 1:
            close_trade(c[i], i, "eow")
        eq_r[i] = equity
        eq_m[i] = equity + (0.0 if pos is None else
                            (c[i] - pos["entry"]) * pos["units"] * pos["dir"])
    return (pd.DataFrame({"equity": eq_r, "equity_mtm": eq_m}, index=times),
            pd.DataFrame(trades))


def evaluate(name, eqs, trs, summ, last_q_folds):
    summary = pd.DataFrame(summ)
    stitched = _stitch(eqs)
    all_tr = pd.concat([t for t in trs if len(t)], ignore_index=True)
    oos = full_report(stitched, all_tr, initial_equity=INIT_EQ,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    passed, detail = _passes_consistency_gate(summary, ret_col="test_return_pct",
                                              pf_col="test_profit_factor",
                                              sharpe_col="test_sharpe_trade")
    lq = [j for j, r in enumerate(summ) if r["fold"] in last_q_folds]
    lq_st = _stitch([eqs[j] for j in lq]) if lq else None
    return {
        "metric": metric_return_over_mtm_dd(stitched, INIT_EQ),
        "return_pct": oos.get("total_return_pct"),
        "max_dd_mtm_pct": oos.get("max_drawdown_mtm_pct"),
        "pf": oos.get("profit_factor"),
        "sharpe_trade": oos.get("sharpe_trade"),
        "n_trades": oos.get("n_trades"),
        "gate_passed": bool(passed), "gate_detail": detail,
        "folds_positive": int((summary.test_return_pct > 0).sum()),
        "n_folds": len(summary),
        "lastq_metric": metric_return_over_mtm_dd(lq_st, INIT_EQ) if lq else None,
        "lastq_return_pct": (float(lq_st["equity"].iloc[-1] / INIT_EQ * 100 - 100)
                             if lq else None),
    }


def main() -> None:
    t0 = time.time()
    m1 = load_mt_ohlcv_csv(Path(CC["bid_path"]), time_col="DateTime", source_tz="UTC",
                           timestamp_is_bar_open=True, bar_duration="1min",
                           start_date=CC["clean_start"])
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(decision, warmup_bars=CFG.warmup_bars,
                                               atr_period=CFG.atr_period,
                                               rsi_period=CFG.rsi_period)
    folds = make_sliding_folds(feat, train_years=CFG.sliding_train_years,
                               val_months=CFG.sliding_val_months,
                               test_months=CFG.sliding_test_months,
                               step_months=CFG.sliding_step_months,
                               embargo_bars=CFG.split_embargo_bars,
                               lockbox_start=CFG.lockbox_start_date)
    n_folds = len(folds)
    q = n_folds // 4
    last_q = list(range(3 * q + 1, n_folds + 1))
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index
    bounds = derive_boundaries(idx)
    emb = CFG.split_embargo_bars

    close = flb["Close"].to_numpy(float)
    atr_all = flb["atr"].to_numpy(float)
    nn = len(flb)
    fwd4 = np.full(nn, np.nan)
    fwd4[: nn - K] = (close[K:] - close[:-K]) / atr_all[: nn - K]
    X_all = flb[feature_cols].to_numpy(float)

    cells = {"V1_5y": ([], [], []), "V1_10y": ([], [], [])}
    for i, (t_0, a, b, c_, d) in enumerate(bounds, start=1):
        te = folds[i - 1][2]
        te_sl = slice(min(c_ + emb, d), d)
        arms = [("V1_5y", slice(a, max(b - emb, a)))]
        t0w = t_0 - pd.DateOffset(months=60)
        if t0w >= idx.min():
            aw = int(idx.searchsorted(t0w, side="left"))
            arms.append(("V1_10y", slice(aw, max(b - emb, aw))))
        for name, sl in arms:
            ytr = fwd4[sl]
            m = ~np.isnan(ytr)
            sc = StandardScaler().fit(X_all[sl][m])
            model = Ridge(alpha=1.0).fit(sc.transform(X_all[sl][m]), ytr[m])
            thr = float(np.quantile(np.abs(model.predict(sc.transform(X_all[sl][m]))), Q))
            s = model.predict(sc.transform(X_all[te_sl]))
            da = (np.sign(s) * (np.abs(s) >= thr)).astype(int)
            eq, tr_df = simulate_v1(te, da)
            rep = full_report(eq, tr_df, initial_equity=INIT_EQ,
                              periods_per_year=CFG.periods_per_year)["value"].to_dict()
            eqs, trs, summ = cells[name]
            eqs.append(eq)
            trs.append(tr_df)
            summ.append({"fold": i,
                         "test_return_pct": rep.get("total_return_pct"),
                         "test_profit_factor": rep.get("profit_factor"),
                         "test_sharpe_trade": rep.get("sharpe_trade"),
                         "fold_metric": metric_return_over_mtm_dd(eq, INIT_EQ)})
        print(f"fold {i:>2}/{n_folds} ({time.time() - t0:,.0f}s)", flush=True)

    report = {"half_cost_atr": HALF, "n_folds": n_folds, "last_quartile_folds": last_q}
    for name, (eqs, trs, summ) in cells.items():
        fold_ids = [r["fold"] for r in summ]
        lq = [f for f in last_q if f in fold_ids]
        res = evaluate(name, eqs, trs, summ, lq)
        report[name] = res
        print(f"\n=== {name} (n={res['n_folds']}): metric {res['metric']:+.4f} | ret "
              f"{res['return_pct']:+.1f}% | PF {res['pf']:.3f} | gate "
              f"{'PASS' if res['gate_passed'] else 'FAIL'} | lastQ "
              f"{res['lastq_return_pct']:+.1f}%")
        for line in res["gate_detail"]:
            print(f"    {line}")
        pd.DataFrame(summ).to_csv(OUT / f"silver_summary_{name}.csv", index=False)
    (OUT / "silver_v1_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nruntime {time.time() - t0:,.0f}s; silver_v1_report.json written")


if __name__ == "__main__":
    main()
