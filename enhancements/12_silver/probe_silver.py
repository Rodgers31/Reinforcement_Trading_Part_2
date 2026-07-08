"""Task 27 Track B step 2 — the FROZEN Task-22 probe on silver alone.

Same 25 features, same labels (k in {2,4,8}; canonical+modal brackets, H=24),
same models/hyperparams, same shuffle controls, same 5y/10y arms — imported
from the committed enhancements/08_probe/probe_edge_ceiling.py. Only the
instrument and its measured cost bar change (silver_census_cost.json).

REPLICATION VERDICT (pinned in Task 27 BEFORE any silver data was seen):
the signal family REPLICATES iff a regression verdict config (fwd4-ridge or
fwd4-HGB) has era-median top-quintile capture >= SILVER'S RT bar in >=2 of 4
eras (eras = consecutive fold quartiles, Task-22 rule). Harness gate first:
shuffle medians within IC +/-0.02, AUC 0.48-0.52. QUARANTINE: nothing here may
change any XAUUSD cell.

Run: .venv/bin/python enhancements/12_silver/probe_silver.py
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
sys.path.insert(0, str(REPO / "enhancements" / "08_probe"))

import probe_edge_ceiling as pr  # noqa: E402  (frozen Task-22 harness)

from config import CFG  # noqa: E402
from data_loader import load_mt_ohlcv_csv, make_sliding_folds, resample_ohlcv  # noqa: E402
from features import prepare_feature_frame  # noqa: E402

OUT = Path(__file__).resolve().parent
CC = json.loads((OUT / "silver_census_cost.json").read_text())
RT = float(CC["rt_cost_atr"])


def era_map(n_folds: int) -> dict:
    q = n_folds // 4
    sizes = [q, q, q, n_folds - 3 * q]
    out, f = {}, 1
    for e, s in enumerate(sizes, start=1):
        for _ in range(s):
            out[f] = e
            f += 1
    return out


def main() -> None:
    t0 = time.time()
    print(f"silver RT bar {RT:.4f} | clean start {CC['clean_start']}")
    m1 = load_mt_ohlcv_csv(Path(CC["bid_path"]), time_col="DateTime", source_tz="UTC",
                           timestamp_is_bar_open=True, bar_duration="1min",
                           start_date=CC["clean_start"])
    decision = resample_ohlcv(m1, CFG.pandas_tf)
    feat, feature_cols = prepare_feature_frame(decision, warmup_bars=CFG.warmup_bars,
                                               atr_period=CFG.atr_period,
                                               rsi_period=CFG.rsi_period)
    assert len(feature_cols) == 25
    folds = make_sliding_folds(feat, train_years=CFG.sliding_train_years,
                               val_months=CFG.sliding_val_months,
                               test_months=CFG.sliding_test_months,
                               step_months=CFG.sliding_step_months,
                               embargo_bars=CFG.split_embargo_bars,
                               lockbox_start=CFG.lockbox_start_date)
    n_folds = len(folds)
    eras = era_map(n_folds)
    print(f"silver folds: {n_folds} | era sizes: "
          f"{pd.Series(eras).value_counts().sort_index().to_dict()}")

    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index
    bounds = pr.derive_boundaries(idx)
    assert len(bounds) == n_folds
    emb = CFG.split_embargo_bars
    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):
        assert flb.iloc[min(c + emb, d):d].index.equals(folds[i - 1][2].index)

    lab = pr.build_labels(flb)
    X_all = flb[feature_cols].to_numpy(float)
    label_cols = [f"fwd{k}" for k in pr.K_VALUES] + \
                 [f"br_{n}_{s}" for n in pr.BRACKETS for s in ("long", "short")]

    rows = []
    for i, (t_0, a, b, c, d) in enumerate(bounds, start=1):
        tr_sl = slice(a, max(b - emb, a))
        te_sl = slice(min(c + emb, d), d)
        arms = [("base5y", tr_sl)]
        t0w = t_0 - pd.DateOffset(months=60)
        if t0w >= idx.min():
            aw = int(idx.searchsorted(t0w, side="left"))
            arms.append(("wide10y", slice(aw, max(b - emb, aw))))
        for arm, sl in arms:
            for lc in label_cols:
                y = lab[lc].to_numpy(float)
                ytr, yte = y[sl], y[te_sl]
                mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
                Xtr, ytr = X_all[sl][mtr], ytr[mtr]
                Xte, yte = X_all[te_sl][mte], yte[mte]
                is_reg = lc.startswith("fwd")
                models = ("ridge", "hgb") if is_reg else ("logit", "hgb")
                kinds = [("real", None)]
                if arm == "base5y":
                    kinds.append(("shuffle", 1000 + i))
                for model_name in models:
                    for kind, seed in kinds:
                        fn = pr.run_regression if is_reg else pr.run_classification
                        met = fn(Xtr, ytr if is_reg else ytr.astype(int),
                                 Xte, yte if is_reg else yte.astype(int),
                                 model_name, shuffle_seed=seed)
                        rows.append(dict(fold=i, era=eras[i], arm=arm, label=lc,
                                         model=model_name, kind=kind, **met))
        print(f"fold {i:>2}/{n_folds} done ({time.time() - t0:,.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "silver_results_per_fold.csv", index=False)

    base = res[(res.arm == "base5y") & (res.kind == "real")]
    shuf = res[res.kind == "shuffle"]
    sh_ic = float(shuf[shuf.label.str.startswith("fwd")]["ic_oos"].median())
    sh_auc = float(shuf[shuf.label.str.startswith("br_")]["auc_oos"].median())
    harness_pass = bool(abs(sh_ic) <= 0.02 and 0.48 <= sh_auc <= 0.52)

    summary = {"rt_cost_atr": RT, "n_folds": n_folds,
               "harness": {"shuffle_ic": sh_ic, "shuffle_auc": sh_auc, "pass": harness_pass},
               "configs": {}}
    replicates = False
    for label, model in [("fwd4", "ridge"), ("fwd4", "hgb"),
                         ("fwd2", "ridge"), ("fwd2", "hgb"),
                         ("fwd8", "ridge"), ("fwd8", "hgb")]:
        d = base[(base.label == label) & (base.model == model)]
        em = d.groupby("era")["capture_q5"].median().to_dict()
        clears = int(sum(v >= RT for v in em.values()))
        entry = {"ic_median": float(d.ic_oos.median()),
                 "ic_pos_folds": int((d.ic_oos > 0).sum()),
                 "era_capture_q5": {str(k): round(float(v), 4) for k, v in em.items()},
                 "eras_clearing": clears}
        summary["configs"][f"{label}|{model}"] = entry
        if label == "fwd4" and clears >= 2:
            replicates = True
    # 10y arm context for fwd4
    w = res[(res.arm == "wide10y") & (res.kind == "real") & (res.label == "fwd4")]
    summary["wide10y_fwd4_era_capture"] = {
        model: {str(k): round(float(v), 4) for k, v in
                w[w.model == model].groupby("era")["capture_q5"].median().items()}
        for model in ("ridge", "hgb")}
    summary["REPLICATES"] = bool(replicates and harness_pass)
    (OUT / "silver_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nREPLICATION VERDICT (pinned rule): "
          f"{'REPLICATES' if summary['REPLICATES'] else 'FAILS TO REPLICATE'}")
    print(f"runtime {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
