"""Task 24 — V3 modal-bracket diagnostic (research-only; pinned in EXECUTION_LOG
Task 24 BEFORE running).

V3 = Task-23 V1 entries (fwd4-ridge top-quintile |score|, train-q80 threshold,
direction = sign) executed under the env's WIDEST bracket: SL 2.0xATR, TP 3R
(= 6xATR from entry), H=24 timeout — every other convention byte-identical to
the Task-23 V2 pin. Sizing at the actual SL distance (0.005*equity/(2*ATR)).
Arms: 5y (folds 1-25), 10y (folds 11-25), plus the 5y folds-11-25 sub-cut.

Purpose: V2 showed the canonical bracket destroys the signal; V3 bounds
whether ANY bracket in the env's menu can monetize the 4-bar drift — pinning
A/B #4's failure attribution (action-space geometry vs optimization).

Pre-flight: recomputes Task-23 V1_5y and V2_5y for fold 11 with the
parameterized simulate() and asserts equality against the committed
summary CSVs (proves the refactor changed nothing) before V3 runs.

Run: .venv/bin/python enhancements/10_ab4/v3_modal_diagnostic.py
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

import run_supervised_baseline as sb  # noqa: E402  (Task-23 module, reused)

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import CFG  # noqa: E402
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402

OUT = Path(__file__).resolve().parent
sb.OUT = OUT  # V3 artifacts (stitched_/summary_/trades_ CSVs) land in 10_ab4/

V3_SL, V3_TP_R = 2.0, 3.0
T23 = REPO / "enhancements" / "09_probe"


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
    assert len(folds) == 25 and CFG.sliding_train_years == 5.0, \
        "V3 diagnostic must run under the anchor config (SLIDING_TRAIN_YEARS unset)"
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index
    bounds = sb.derive_boundaries(idx)
    emb = CFG.split_embargo_bars
    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):
        assert flb.iloc[min(c + emb, d):d].index.equals(folds[i - 1][2].index)

    close = flb["Close"].to_numpy(float)
    atr_all = flb["atr"].to_numpy(float)
    nn = len(flb)
    fwd4 = np.full(nn, np.nan)
    fwd4[: nn - sb.K] = (close[sb.K:] - close[:-sb.K]) / atr_all[: nn - sb.K]
    X_all = flb[feature_cols].to_numpy(float)

    def fit_scores(sl_tr, te_sl):
        ytr = fwd4[sl_tr]
        m = ~np.isnan(ytr)
        sc = StandardScaler().fit(X_all[sl_tr][m])
        model = Ridge(alpha=1.0).fit(sc.transform(X_all[sl_tr][m]), ytr[m])
        thr = float(np.quantile(np.abs(model.predict(sc.transform(X_all[sl_tr][m]))), sb.Q))
        return model.predict(sc.transform(X_all[te_sl])), thr

    # ── pre-flight: parameterized simulate() reproduces committed V1/V2 (fold 11, 5y)
    t_0, a, b, c, d = bounds[10]
    te = folds[10][2]
    te_sl = slice(min(c + emb, d), d)
    scores, thr = fit_scores(slice(a, max(b - emb, a)), te_sl)
    for variant, csv in (("V1", "summary_V1_5y.csv"), ("V2", "summary_V2_5y.csv")):
        eq, tr_df, _ = sb.simulate(te, scores, thr, variant)   # canonical defaults
        row_new = sb.fold_row(11, eq, tr_df)
        row_old = pd.read_csv(T23 / csv).query("fold == 11").iloc[0].to_dict()
        for key in ("test_return_pct", "test_profit_factor", "fold_metric", "test_n_trades"):
            assert abs(float(row_new[key]) - float(row_old[key])) < 1e-9, \
                f"pre-flight FAIL: {variant} fold 11 {key} {row_new[key]} != {row_old[key]}"
    print("[OK] pre-flight: parameterized simulate() reproduces committed V1/V2 fold-11 rows")

    # ── V3 cells
    store = {"V3_5y": ([], [], []), "V3_10y": ([], [], [])}
    diags = {k: [] for k in store}
    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):
        te = folds[i - 1][2]
        te_sl = slice(min(c + emb, d), d)
        arms = [("V3_5y", slice(a, max(b - emb, a)))]
        if i in sb.W10_FOLDS:
            aw = int(idx.searchsorted(t_0 - pd.DateOffset(months=60), side="left"))
            arms.append(("V3_10y", slice(aw, max(b - emb, aw))))
        for name, sl_tr in arms:
            scores, thr = fit_scores(sl_tr, te_sl)
            eq, tr_df, dg = sb.simulate(te, scores, thr, "V3",
                                        sl_mult=V3_SL, tp_r=V3_TP_R)
            eqs, trs, summ = store[name]
            eqs.append(eq)
            trs.append(tr_df)
            summ.append(sb.fold_row(i, eq, tr_df))
            dg.update(fold=i)
            diags[name].append(dg)
        print(f"fold {i:>2}/25 ({time.time() - t0:,.0f}s)", flush=True)

    results = {"pinned": {"sl_mult": V3_SL, "tp_r": V3_TP_R, "H": sb.H,
                          "entries": "Task-23 V1 (fwd4-ridge top-quintile)"}}
    for name, (eqs, trs, summ) in store.items():
        fold_ids = [r["fold"] for r in summ]
        res = sb.evaluate_variant(name, eqs, trs, summ, fold_ids)
        res["diag"] = {
            "ambiguous_both_touch_total": int(sum(g["ambiguous_both_touch"] for g in diags[name])),
            "skipped_in_pos_total": int(sum(g["skipped_in_pos"] for g in diags[name])),
        }
        results[name] = res
    # like-for-like 5y sub-cut on folds 11-25
    eqs, trs, summ = store["V3_5y"]
    keep = [j for j, r in enumerate(summ) if r["fold"] in sb.W10_FOLDS]
    results["V3_5y_f11_25"] = sb.evaluate_variant(
        "V3_5y_f11_25", [eqs[j] for j in keep], [trs[j] for j in keep],
        [summ[j] for j in keep], sb.W10_FOLDS)

    (OUT / "v3_report.json").write_text(json.dumps(results, indent=2, default=float))
    for name in ("V3_5y", "V3_10y", "V3_5y_f11_25"):
        x = results[name]
        print(f"{name:>13}: metric {x['metric_stitched']:+.4f} ret {x['stitched_return_pct']:+.1f}% "
              f"PF {x['stitched_pf']:.3f} gate {'PASS' if x['gate_passed'] else 'FAIL'} "
              f"E4m {x['E4_metric_stitched']:+.4f} E4ret {x['E4_return_pct']:+.1f}% "
              f"exits {x['exit_reason_mix']}")
    print(f"\nruntime {time.time() - t0:,.0f}s; v3_report.json written")


if __name__ == "__main__":
    main()
