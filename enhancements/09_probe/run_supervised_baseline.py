"""Task 23 — SUPERVISED BASELINE THROUGH THE HONEST RULER (research-only).

Protocol pinned in enhancements/EXECUTION_LOG.md Task 23 (commit 6f6db2a) BEFORE
this script ran. Two pre-declared variants, no parameter search:

  V1: fwd4-ridge score, trade top-quintile |score| bars (train-derived q80
      threshold), direction = sign(score), exit at the close of the 4th bar.
  V2: same entries, canonical bracket SL 1.0xATR / TP 1R, H=24 H1-touch scan,
      gap-through at open, both-touch -> SL first (ambiguity counted).

Mechanics mirror env_bracket exactly: entry close +/- half_cost, exit raw -/+
half_cost (both legs at ENTRY-bar ATR), commission $0.01/trade, fixed-
fractional sizing units = 0.005*equity/(1.0*ATR_entry), one position, no flips.

Ruler = the anchor's own functions, unmodified: evaluate.full_report,
run_baseline._stitch, eval_harness.metric_return_over_mtm_dd,
train_ppo._passes_consistency_gate. Anchor per-seed sub-stitches (folds 11-25,
19-25) recomputed from the anchor run's own test_equity.csv files; the full
25-fold re-stitch is asserted against baseline_report.json as a cross-check.

Run from repo root: .venv/bin/python enhancements/09_probe/run_supervised_baseline.py
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
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features, _passes_consistency_gate  # noqa: E402
from run_baseline import _stitch  # noqa: E402
from eval_harness import metric_return_over_mtm_dd  # noqa: E402
from evaluate import full_report  # noqa: E402

OUT = Path(__file__).resolve().parent
ANCHOR_RUN = REPO / "runs" / "20260703-021710_b9bc9d6_baseline-3seed"
ANCHOR_SEEDS = [42, 43, 44, 45, 46]

K = 4
Q = 0.8                                   # top-quintile |score| threshold (train)
H = 24                                    # V2 horizon (bars)
SL_MULT, TP_R = 1.0, 1.0                  # canonical bracket
HALF_COST_ATR = CFG.spread_atr_frac / 2.0 + CFG.slippage_atr_frac   # 0.03415
COMMISSION = CFG.commission_per_trade     # 0.01
RISK_FRAC = CFG.risk_fraction             # 0.005
INIT_EQ = CFG.initial_equity              # 10_000
ERA = {f: (1 if f <= 6 else 2 if f <= 12 else 3 if f <= 18 else 4) for f in range(1, 26)}
E4_FOLDS = list(range(19, 26))
W10_FOLDS = list(range(11, 26))


def derive_boundaries(idx):
    """Same re-derivation validated in Task 22 (asserted vs make_sliding_folds)."""
    start, end = idx.min(), idx.max()
    train_off = pd.DateOffset(months=int(round(CFG.sliding_train_years * 12)))
    val_off = pd.DateOffset(months=CFG.sliding_val_months)
    test_off = pd.DateOffset(months=CFG.sliding_test_months)
    step_off = pd.DateOffset(months=CFG.sliding_step_months)
    out, t0 = [], start
    while True:
        val_start = t0 + train_off
        test_start = val_start + val_off
        test_end = test_start + test_off
        if test_end > end:
            break
        out.append((t0,
                    int(idx.searchsorted(t0, side="left")),
                    int(idx.searchsorted(val_start, side="left")),
                    int(idx.searchsorted(test_start, side="left")),
                    int(idx.searchsorted(test_end, side="left"))))
        t0 = t0 + step_off
        if t0 + train_off >= end:
            break
    return out


def simulate(te: pd.DataFrame, scores: np.ndarray, thr: float, variant: str,
             sl_mult: float = SL_MULT, tp_r: float = TP_R):
    """One test window, one variant. Returns (equity_df, trades_df, diag).

    sl_mult/tp_r default to the canonical bracket (Task-23 pin) so V1/V2 cells
    are byte-identical to the committed run; the Task-24 V3 diagnostic passes
    the modal bracket (2.0, 3.0). Sizing always uses the ACTUAL SL distance
    (env formula)."""
    o = te["Open"].to_numpy(float)
    h = te["High"].to_numpy(float)
    l = te["Low"].to_numpy(float)
    c = te["Close"].to_numpy(float)
    atr = te["atr"].to_numpy(float)
    times = te.index
    n = len(te)
    selected = np.abs(scores) >= thr

    equity = INIT_EQ
    pos = None
    eq_r = np.empty(n)
    eq_m = np.empty(n)
    trades = []
    n_selected_bars = int(selected.sum())
    skipped_in_pos = 0
    ambig = 0

    def close_trade(raw, i, reason):
        nonlocal equity, pos
        exit_price = raw - pos["dir"] * HALF_COST_ATR * pos["atr_e"]
        pnl = (exit_price - pos["entry"]) * pos["units"] * pos["dir"] - COMMISSION
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
        # 1) exits
        if pos is not None:
            d = pos["dir"]
            if variant == "V1":
                if i - pos["i"] >= K:
                    close_trade(c[i], i, "time")
            else:
                if i > pos["i"]:
                    sl, tp = pos["sl"], pos["tp"]
                    raw = reason = None
                    if (d == 1 and o[i] <= sl) or (d == -1 and o[i] >= sl):
                        raw, reason = o[i], "SL_gap"
                    elif (d == 1 and o[i] >= tp) or (d == -1 and o[i] <= tp):
                        raw, reason = o[i], "TP_gap"
                    else:
                        sl_t = (l[i] <= sl) if d == 1 else (h[i] >= sl)
                        tp_t = (h[i] >= tp) if d == 1 else (l[i] <= tp)
                        if sl_t and tp_t:
                            raw, reason = sl, "SL"
                            ambig += 1
                        elif sl_t:
                            raw, reason = sl, "SL"
                        elif tp_t:
                            raw, reason = tp, "TP"
                    if raw is not None:
                        close_trade(raw, i, reason)
                    elif i - pos["i"] >= H:
                        close_trade(c[i], i, "timeout")
        # 2) entry (env allows a final-bar entry; the eow close below realizes it)
        if pos is None and selected[i]:
            d = 1 if scores[i] > 0 else -1
            atr_e = max(atr[i], 1e-12)
            entry = c[i] + d * HALF_COST_ATR * atr_e
            risk = max(equity * RISK_FRAC, 1e-8)
            units = risk / (sl_mult * atr_e)
            pos = {"dir": d, "entry": entry, "mid": c[i], "units": units,
                   "risk": risk, "atr_e": atr_e, "i": i, "t": times[i],
                   "sl": entry - d * sl_mult * atr_e,
                   "tp": entry + d * tp_r * sl_mult * atr_e}
        elif pos is not None and selected[i] and i != pos["i"]:
            skipped_in_pos += 1
        # 3) window end
        if pos is not None and i == n - 1:
            close_trade(c[i], i, "eow")
        # 4) mark
        eq_r[i] = equity
        eq_m[i] = equity + (0.0 if pos is None else
                            (c[i] - pos["entry"]) * pos["units"] * pos["dir"])

    eq_df = pd.DataFrame({"equity": eq_r, "equity_mtm": eq_m}, index=times)
    tr_df = pd.DataFrame(trades)
    diag = {"n_selected_bars": n_selected_bars, "skipped_in_pos": skipped_in_pos,
            "ambiguous_both_touch": ambig, "n_bars": n}
    return eq_df, tr_df, diag


def fold_row(fold, eq, tr):
    rep = full_report(eq, tr, initial_equity=INIT_EQ,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    return {
        "fold": fold, "era": ERA[fold],
        "test_return_pct": rep.get("total_return_pct"),
        "test_profit_factor": rep.get("profit_factor"),
        "test_sharpe_trade": rep.get("sharpe_trade"),
        "test_max_dd_mtm_pct": rep.get("max_drawdown_mtm_pct"),
        "test_n_trades": rep.get("n_trades"),
        "win_rate_pct": rep.get("win_rate_pct"),
        "fold_metric": metric_return_over_mtm_dd(eq, INIT_EQ),
    }


def evaluate_variant(name, eqs, trs, summaries, fold_ids):
    """Stitch + metric + gate + E4 subset for one (variant, arm)."""
    summary = pd.DataFrame(summaries)
    stitched = _stitch(eqs)
    all_tr = pd.concat([t for t in trs if len(t)], ignore_index=True) \
        if any(len(t) for t in trs) else pd.DataFrame()
    oos = full_report(stitched, all_tr, initial_equity=INIT_EQ,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    passed, detail = _passes_consistency_gate(
        summary, ret_col="test_return_pct", pf_col="test_profit_factor",
        sharpe_col="test_sharpe_trade")
    e4_idx = [j for j, f in enumerate(fold_ids) if f in E4_FOLDS]
    e4_stitch = _stitch([eqs[j] for j in e4_idx]) if e4_idx else None
    exit_mix = (all_tr["exit_reason"].value_counts().to_dict() if len(all_tr) else {})
    gross = float(all_tr["gross_pnl_mid"].sum()) if len(all_tr) else 0.0
    cost = float(all_tr["cost_paid"].sum()) if len(all_tr) else 0.0
    out = {
        "metric_stitched": metric_return_over_mtm_dd(stitched, INIT_EQ),
        "stitched_return_pct": oos.get("total_return_pct"),
        "stitched_max_dd_mtm_pct": oos.get("max_drawdown_mtm_pct"),
        "stitched_pf": oos.get("profit_factor"),
        "stitched_sharpe_trade": oos.get("sharpe_trade"),
        "n_trades": oos.get("n_trades"),
        "win_rate_pct": oos.get("win_rate_pct"),
        "avg_bars_in_trade": oos.get("avg_bars_in_trade"),
        "gate_passed": bool(passed),
        "gate_detail": detail,
        "folds_positive": int((summary.test_return_pct > 0).sum()),
        "n_folds": len(summary),
        "era_fold_metric_median": {str(e): float(m) for e, m in
                                   summary.groupby("era").fold_metric.median().items()},
        "era_return_median": {str(e): float(m) for e, m in
                              summary.groupby("era").test_return_pct.median().items()},
        "E4_metric_stitched": (metric_return_over_mtm_dd(e4_stitch, INIT_EQ)
                               if e4_stitch is not None else None),
        "E4_return_pct": (float(e4_stitch["equity"].iloc[-1] / INIT_EQ * 100 - 100)
                          if e4_stitch is not None and len(e4_stitch) else None),
        "exit_reason_mix": exit_mix,
        "gross_pnl_mid_total": gross,
        "cost_paid_total": cost,
        "cost_over_gross_pct": (100.0 * cost / gross if gross > 0 else None),
    }
    stitched.to_csv(OUT / f"stitched_{name}.csv.gz", compression="gzip")
    summary.to_csv(OUT / f"summary_{name}.csv", index=False)
    if len(all_tr):
        all_tr.to_csv(OUT / f"trades_{name}.csv.gz", index=False, compression="gzip")
    return out


def anchor_substitch():
    """Anchor per-seed metrics re-stitched over fold subsets, + full-25 cross-check."""
    ref = json.loads((ANCHOR_RUN / "baseline_report.json").read_text())
    ref_per_seed = {int(k): v for k, v in ref["metric_per_seed"].items()}
    out = {}
    for subset_name, fold_ids in [("full25", list(range(1, 26))),
                                  ("f11_25", W10_FOLDS), ("f19_25", E4_FOLDS)]:
        per_seed = {}
        for s in ANCHOR_SEEDS:
            eqs = [pd.read_csv(ANCHOR_RUN / "jobs" / f"f{k:02d}_s{s}" / "test_equity.csv",
                               index_col=0) for k in fold_ids]
            per_seed[s] = metric_return_over_mtm_dd(_stitch(eqs), INIT_EQ)
        out[subset_name] = {"per_seed": {str(s): round(v, 4) for s, v in per_seed.items()},
                            "median": float(np.median(list(per_seed.values())))}
    for s in ANCHOR_SEEDS:  # cross-check the ruler reuse against the ratified report
        mine, theirs = out["full25"]["per_seed"][str(s)], ref_per_seed[s]
        assert abs(mine - theirs) < 5e-4, f"anchor re-stitch mismatch seed {s}: {mine} vs {theirs}"
    return out


def main():
    t0 = time.time()
    print("Loading data (anchor-identical path)…")
    _, feat, feature_cols = _load_decision_features()
    assert len(feature_cols) == 25
    folds = make_sliding_folds(feat, train_years=CFG.sliding_train_years,
                               val_months=CFG.sliding_val_months,
                               test_months=CFG.sliding_test_months,
                               step_months=CFG.sliding_step_months,
                               embargo_bars=CFG.split_embargo_bars,
                               lockbox_start=CFG.lockbox_start_date)
    assert len(folds) == 25
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index
    bounds = derive_boundaries(idx)
    emb = CFG.split_embargo_bars
    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):  # revalidate vs library
        tr, va, te = folds[i - 1]
        assert flb.iloc[a:max(b - emb, a)].index.equals(tr.index)
        assert flb.iloc[min(c + emb, d):d].index.equals(te.index)
    print("25/25 fold boundaries revalidated.")

    close = flb["Close"].to_numpy(float)
    atr_all = flb["atr"].to_numpy(float)
    nn = len(flb)
    fwd4 = np.full(nn, np.nan)
    fwd4[: nn - K] = (close[K:] - close[:-K]) / atr_all[: nn - K]
    X_all = flb[feature_cols].to_numpy(float)

    anchor = anchor_substitch()
    print(f"Anchor re-stitch cross-check PASS. full25 median "
          f"{anchor['full25']['median']:+.4f} | f11-25 {anchor['f11_25']['median']:+.4f} "
          f"| f19-25 {anchor['f19_25']['median']:+.4f}")

    results = {}
    store = {("V1", "5y"): ([], [], []), ("V2", "5y"): ([], [], []),
             ("V1", "10y"): ([], [], []), ("V2", "10y"): ([], [], [])}
    diags = {k: [] for k in store}

    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):
        te = folds[i - 1][2]
        te_sl = slice(min(c + emb, d), d)
        arms = [("5y", slice(a, max(b - emb, a)))]
        if i in W10_FOLDS:
            t0w = t_0 - pd.DateOffset(months=60)
            assert t0w >= idx.min()
            aw = int(idx.searchsorted(t0w, side="left"))
            arms.append(("10y", slice(aw, max(b - emb, aw))))
        for arm, sl_tr in arms:
            ytr = fwd4[sl_tr]
            m = ~np.isnan(ytr)
            Xtr = X_all[sl_tr][m]
            sc = StandardScaler().fit(Xtr)
            model = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr[m])
            thr = float(np.quantile(np.abs(model.predict(sc.transform(Xtr))), Q))
            scores = model.predict(sc.transform(X_all[te_sl]))
            for variant in ("V1", "V2"):
                eq, tr_df, dg = simulate(te, scores, thr, variant)
                eqs, trs, summ = store[(variant, arm)]
                eqs.append(eq)
                trs.append(tr_df)
                summ.append(fold_row(i, eq, tr_df))
                dg.update(fold=i)
                diags[(variant, arm)].append(dg)
        print(f"fold {i:>2}/25 simulated ({time.time() - t0:,.0f}s)", flush=True)

    report = {"anchor": anchor,
              "pinned": {"rt_cost_atr_frac": round(2 * HALF_COST_ATR, 4),
                         "commission": COMMISSION, "risk_frac": RISK_FRAC,
                         "k": K, "H": H, "bracket": [SL_MULT, TP_R], "q": Q}}
    for (variant, arm), (eqs, trs, summ) in store.items():
        fold_ids = [r["fold"] for r in summ]
        name = f"{variant}_{arm}"
        res = evaluate_variant(name, eqs, trs, summ, fold_ids)
        res["diag"] = {
            "selected_bar_frac": round(float(np.mean(
                [g["n_selected_bars"] / g["n_bars"] for g in diags[(variant, arm)]])), 4),
            "skipped_in_pos_total": int(sum(g["skipped_in_pos"] for g in diags[(variant, arm)])),
            "ambiguous_both_touch_total": int(sum(g["ambiguous_both_touch"]
                                                  for g in diags[(variant, arm)])),
        }
        results[name] = res
        print(f"\n=== {name}: metric {res['metric_stitched']:+.4f} | return "
              f"{res['stitched_return_pct']:+.1f}% | maxDD_mtm {res['stitched_max_dd_mtm_pct']:.1f}% "
              f"| PF {res['stitched_pf']:.3f} | trades {res['n_trades']} | gate "
              f"{'PASS' if res['gate_passed'] else 'FAIL'} | E4 metric "
              f"{res['E4_metric_stitched'] if res['E4_metric_stitched'] is not None else float('nan'):+.4f}")
        for line in res["gate_detail"]:
            print(f"    {line}")
    # 5y arm restricted to folds 11-25 for the like-for-like 10y comparison
    for variant in ("V1", "V2"):
        eqs, trs, summ = store[(variant, "5y")]
        keep = [j for j, r in enumerate(summ) if r["fold"] in W10_FOLDS]
        res = evaluate_variant(f"{variant}_5y_f11_25",
                               [eqs[j] for j in keep], [trs[j] for j in keep],
                               [summ[j] for j in keep], W10_FOLDS)
        results[f"{variant}_5y_f11_25"] = res

    report["results"] = results
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nTotal runtime {time.time() - t0:,.0f}s. report.json written.")


if __name__ == "__main__":
    main()
