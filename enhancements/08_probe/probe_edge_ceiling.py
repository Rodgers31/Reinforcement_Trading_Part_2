"""Task 22 — SUPERVISED EDGE-CEILING PROBE (research-only).

Protocol pinned in enhancements/EXECUTION_LOG.md Task 22 (commit 209179b) BEFORE
any label or metric was computed. Everything here follows that pin:

- Anchor-identical data path: train_ppo._load_decision_features() -> 25 sliding
  folds via data_loader.make_sliding_folds (5y/6m/6m/6m, embargo 200, lockbox
  2024-07-01). Fold boundaries re-derived locally ONLY to obtain integer cut
  points for the 10y widening arm, and asserted index-identical to the library's.
- Inputs: the 25 market feature columns the env feeds PPO. Fits on TRAIN only,
  metrics on TEST (val untouched).
- Labels: k-bar ATR-normalized forward return, k in {2,4,8}; bracket-aligned
  TP-before-SL within H=24 bars for canonical (SL 1.0xATR, TP 1R) and
  modal-anchor (SL 2.0xATR, TP 3R) brackets, long and short, mid-price
  brackets, same-bar-both-touch -> SL first, no-touch -> 0.
- Models: Ridge / LogisticRegression (train-standardized) + sklearn
  HistGradientBoosting (fixed hyperparameters, zero tuning).
- Controls: per (fold x config) shuffled-train-label refit, seed=1000+fold.
- NO training-run side effects: no registry row, no INDEX row, anchor untouched.

Run from repo root:  .venv/bin/python enhancements/08_probe/probe_edge_ceiling.py
Outputs (this directory): fold_windows.csv, results_per_fold.csv,
results_univariate.csv, summary.json, probe_stdout.log (via tee in launcher).
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

from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import CFG  # noqa: E402
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# ── Pinned constants (EXECUTION_LOG Task 22) ────────────────────────────────
K_VALUES = (2, 4, 8)                 # anchor bars_in_trade p25/p50/p75
PRIMARY_K = 4
BRACKET_H = 24                       # bars
BRACKETS = {                         # name -> (sl_atr_mult, tp_r)
    "canon": (1.0, 1.0),             # PRIMARY
    "modal": (2.0, 3.0),             # anchor's most-chosen (14.4%)
}
RT_COST_ATR = CFG.spread_atr_frac + 2.0 * CFG.slippage_atr_frac  # 0.0683
ERAS = {1: range(1, 7), 2: range(7, 13), 3: range(13, 19), 4: range(19, 26)}
HGB_KW = dict(max_iter=200, learning_rate=0.08, min_samples_leaf=50,
              l2_regularization=1.0, early_stopping=False, random_state=0)


def era_of(fold: int) -> int:
    for e, rng in ERAS.items():
        if fold in rng:
            return e
    raise ValueError(fold)


# ── Fold boundary re-derivation (mirrors make_sliding_folds exactly) ────────
def derive_boundaries(idx: pd.DatetimeIndex):
    """Integer (a, b, c, d) cut points per fold, replicating make_sliding_folds."""
    start, end = idx.min(), idx.max()
    train_off = pd.DateOffset(months=int(round(CFG.sliding_train_years * 12)))
    val_off = pd.DateOffset(months=CFG.sliding_val_months)
    test_off = pd.DateOffset(months=CFG.sliding_test_months)
    step_off = pd.DateOffset(months=CFG.sliding_step_months)
    out = []
    t0 = start
    while True:
        val_start = t0 + train_off
        test_start = val_start + val_off
        test_end = test_start + test_off
        if test_end > end:
            break
        a = int(idx.searchsorted(t0, side="left"))
        b = int(idx.searchsorted(val_start, side="left"))
        c = int(idx.searchsorted(test_start, side="left"))
        d = int(idx.searchsorted(test_end, side="left"))
        out.append((t0, a, b, c, d))
        t0 = t0 + step_off
        if t0 + train_off >= end:
            break
    return out


# ── Label construction (on the lockbox-truncated frame) ─────────────────────
def build_labels(feat: pd.DataFrame) -> pd.DataFrame:
    close = feat["Close"].to_numpy(float)
    high = feat["High"].to_numpy(float)
    low = feat["Low"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)
    n = len(feat)
    lab = pd.DataFrame(index=feat.index)

    for k in K_VALUES:
        fwd = np.full(n, np.nan)
        fwd[: n - k] = (close[k:] - close[:-k]) / atr[: n - k]
        lab[f"fwd{k}"] = fwd

    inf = np.iinfo(np.int32).max
    for name, (sl_mult, tp_r) in BRACKETS.items():
        sl_d = sl_mult * atr
        tp_d = tp_r * sl_d
        for side, sgn in (("long", 1.0), ("short", -1.0)):
            tp_price = close + sgn * tp_d
            sl_price = close - sgn * sl_d
            first_tp = np.full(n, inf, dtype=np.int64)
            first_sl = np.full(n, inf, dtype=np.int64)
            for j in range(1, BRACKET_H + 1):
                hi_j = np.full(n, np.nan)
                lo_j = np.full(n, np.nan)
                hi_j[: n - j] = high[j:]
                lo_j[: n - j] = low[j:]
                if sgn > 0:
                    tp_touch = hi_j >= tp_price
                    sl_touch = lo_j <= sl_price
                else:
                    tp_touch = lo_j <= tp_price
                    sl_touch = hi_j >= sl_price
                first_tp = np.where(tp_touch & (first_tp == inf), j, first_tp)
                first_sl = np.where(sl_touch & (first_sl == inf), j, first_sl)
            # TP strictly first wins; ties (same bar) -> SL first (env convention);
            # neither touched -> 0.
            lab[f"br_{name}_{side}"] = (first_tp < first_sl).astype(int)
    return lab


# ── Metric helpers ───────────────────────────────────────────────────────────
def s_ic(pred, y):
    if len(y) < 10 or np.all(pred == pred[0]):
        return np.nan
    return float(spearmanr(pred, y).statistic)


def capture(pred, y):
    s = np.sign(pred)
    m = s != 0
    if m.sum() < 10:
        return np.nan
    return float(np.mean(y[m] * s[m]))


def run_regression(Xtr, ytr, Xte, yte, model_name, shuffle_seed=None):
    if shuffle_seed is not None:
        ytr = np.random.default_rng(shuffle_seed).permutation(ytr)
    if model_name == "ridge":
        sc = StandardScaler().fit(Xtr)
        model = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        ptr, pte = model.predict(sc.transform(Xtr)), model.predict(sc.transform(Xte))
    else:
        model = HistGradientBoostingRegressor(**HGB_KW).fit(Xtr, ytr)
        ptr, pte = model.predict(Xtr), model.predict(Xte)
    thr = np.quantile(np.abs(ptr), 0.8)          # train-derived, no test peeking
    sel = np.abs(pte) >= thr
    return {
        "ic_train": s_ic(ptr, ytr),
        "ic_oos": s_ic(pte, yte),
        "capture_all": capture(pte, yte),
        "capture_q5": capture(pte[sel], yte[sel]) if sel.sum() >= 10 else np.nan,
        "q5_frac": float(sel.mean()),
    }


def run_classification(Xtr, ytr, Xte, yte, model_name, shuffle_seed=None):
    if shuffle_seed is not None:
        ytr = np.random.default_rng(shuffle_seed).permutation(ytr)
    if len(np.unique(ytr)) < 2:
        return {k: np.nan for k in ("auc_train", "auc_oos", "base_rate_test",
                                    "topdec_tp_rate", "topdec_frac")}
    if model_name == "logit":
        sc = StandardScaler().fit(Xtr)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr), ytr)
        ptr = model.predict_proba(sc.transform(Xtr))[:, 1]
        pte = model.predict_proba(sc.transform(Xte))[:, 1]
    else:
        model = HistGradientBoostingClassifier(**HGB_KW).fit(Xtr, ytr)
        ptr = model.predict_proba(Xtr)[:, 1]
        pte = model.predict_proba(Xte)[:, 1]
    thr = np.quantile(ptr, 0.9)                  # train-derived
    sel = pte >= thr
    auc_tr = roc_auc_score(ytr, ptr)
    auc_te = roc_auc_score(yte, pte) if len(np.unique(yte)) > 1 else np.nan
    return {
        "auc_train": float(auc_tr),
        "auc_oos": float(auc_te) if auc_te == auc_te else np.nan,
        "base_rate_test": float(np.mean(yte)),
        "topdec_tp_rate": float(np.mean(yte[sel])) if sel.sum() >= 10 else np.nan,
        "topdec_frac": float(sel.mean()),
    }


def main():
    t_start = time.time()
    print(f"RT cost bar (ATR units): {RT_COST_ATR:.4f}")
    p_star = {name: (1.0 + RT_COST_ATR / BRACKETS[name][0]) /
                    (1.0 + BRACKETS[name][1])
              for name in BRACKETS}
    # p* = (b + c)/(a + b) with a = tp_r*b, b = sl_mult*ATR, c = RT*ATR
    #    = (sl_mult + RT)/(sl_mult*(1+tp_r)) = (1 + RT/sl_mult)/(1+tp_r)
    for name in BRACKETS:
        print(f"p* {name}: {p_star[name]:.4f}")

    print("Loading data (anchor-identical path)…")
    _, feat, feature_cols = _load_decision_features()
    assert len(feature_cols) == 25, f"expected 25 features, got {len(feature_cols)}"

    folds = make_sliding_folds(
        feat,
        train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months,
        test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months,
        embargo_bars=CFG.split_embargo_bars,
        lockbox_start=CFG.lockbox_start_date,
    )
    assert len(folds) == 25, f"expected the anchor's 25 folds, got {len(folds)}"

    # Lockbox-truncated frame: labels can never touch lockbox bars.
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    flb = feat.loc[feat.index < cut].copy()
    idx = flb.index

    bounds = derive_boundaries(idx)
    assert len(bounds) == 25
    emb = CFG.split_embargo_bars
    windows = []
    for i, (t0, a, b, c, d) in enumerate(bounds, start=1):
        tr = flb.iloc[a:max(b - emb, a)]
        va = flb.iloc[min(b + emb, c):c]
        te = flb.iloc[min(c + emb, d):d]
        ltr, lva, lte = folds[i - 1]
        assert tr.index.equals(ltr.index) and va.index.equals(lva.index) \
            and te.index.equals(lte.index), f"fold {i} boundary mismatch"
        # 10y widening arm: pull t0 back 5y; qualifies if data exists.
        t0w = t0 - pd.DateOffset(months=60)
        wide_ok = t0w >= idx.min()
        aw = int(idx.searchsorted(t0w, side="left")) if wide_ok else None
        windows.append(dict(fold=i, era=era_of(i), a=a, b=b, c=c, d=d,
                            aw=aw, wide_ok=wide_ok,
                            train_start=str(tr.index.min()), train_end=str(tr.index.max()),
                            test_start=str(te.index.min()), test_end=str(te.index.max()),
                            n_train=len(tr), n_test=len(te)))
    print(f"All 25 re-derived fold boundaries match make_sliding_folds. "
          f"Widening arm qualifies: {sum(w['wide_ok'] for w in windows)} folds.")
    pd.DataFrame(windows).drop(columns=["a", "b", "c", "d", "aw"]).to_csv(
        OUT_DIR / "fold_windows.csv", index=False)

    print("Building labels…")
    lab = build_labels(flb)
    X_all = flb[feature_cols].to_numpy(float)

    label_cols = [f"fwd{k}" for k in K_VALUES] + \
                 [f"br_{n}_{s}" for n in BRACKETS for s in ("long", "short")]

    rows = []
    uni_rows = []
    for w in windows:
        fold, a, b, c, d = w["fold"], w["a"], w["b"], w["c"], w["d"]
        tr_sl = slice(a, max(b - emb, a))
        te_sl = slice(min(c + emb, d), d)
        arms = [("base5y", tr_sl)]
        if w["wide_ok"]:
            arms.append(("wide10y", slice(w["aw"], max(b - emb, w["aw"]))))

        # Univariate OOS screen (k=4), diagnostic only.
        y4te = lab[f"fwd{PRIMARY_K}"].to_numpy(float)[te_sl]
        m4 = ~np.isnan(y4te)
        for fi, fname in enumerate(feature_cols):
            uni_rows.append(dict(fold=fold, era=w["era"], feature=fname,
                                 ic_oos=s_ic(X_all[te_sl][m4][:, fi], y4te[m4])))

        for arm, sl in arms:
            Xtr_full = X_all[sl]
            Xte_full = X_all[te_sl]
            for lc in label_cols:
                y = lab[lc].to_numpy(float)
                ytr, yte = y[sl], y[te_sl]
                mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
                Xtr, ytr = Xtr_full[mtr], ytr[mtr]
                Xte, yte = Xte_full[mte], yte[mte]
                is_reg = lc.startswith("fwd")
                models = ("ridge", "hgb") if is_reg else ("logit", "hgb")
                kinds = [("real", None)]
                if arm == "base5y":
                    kinds.append(("shuffle", 1000 + fold))
                for model_name in models:
                    for kind, seed in kinds:
                        fn = run_regression if is_reg else run_classification
                        met = fn(Xtr, ytr if is_reg else ytr.astype(int),
                                 Xte, yte if is_reg else yte.astype(int),
                                 model_name, shuffle_seed=seed)
                        rows.append(dict(fold=fold, era=w["era"], arm=arm,
                                         label=lc, model=model_name, kind=kind,
                                         n_train=len(ytr), n_test=len(yte), **met))
        done = len([r for r in rows if r["fold"] == fold])
        print(f"fold {fold:>2}/25 done ({done} rows, "
              f"{time.time() - t_start:,.0f}s elapsed)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "results_per_fold.csv", index=False)
    uni = pd.DataFrame(uni_rows)
    uni.to_csv(OUT_DIR / "results_univariate.csv", index=False)

    # ── Aggregation + pinned verdict rules ──────────────────────────────────
    base = res[(res.arm == "base5y") & (res.kind == "real")]
    shuf = res[(res.kind == "shuffle")]

    def era_median(df, col):
        return df.groupby("era")[col].median().to_dict()

    summary = {"rt_cost_atr": RT_COST_ATR, "p_star": p_star,
               "harness": {}, "verdict_configs": {}, "widening": {}}

    # Harness-integrity gate.
    sh_ic = shuf[shuf.label.str.startswith("fwd")]["ic_oos"].median()
    sh_auc = shuf[shuf.label.str.startswith("br_")]["auc_oos"].median()
    summary["harness"] = {"shuffle_median_ic": float(sh_ic),
                          "shuffle_median_auc": float(sh_auc),
                          "pass": bool(abs(sh_ic) <= 0.02 and 0.48 <= sh_auc <= 0.52)}

    # Verdict configs (pinned).
    vcfg = [("fwd4", "ridge"), ("fwd4", "hgb"),
            ("br_canon_long", "logit"), ("br_canon_long", "hgb"),
            ("br_canon_short", "logit"), ("br_canon_short", "hgb")]
    any_sign, any_clear = False, False
    for label, model in vcfg:
        d = base[(base.label == label) & (base.model == model)]
        entry = {"n_folds": int(len(d))}
        if label.startswith("fwd"):
            entry["ic_median"] = float(d.ic_oos.median())
            entry["ic_pos_folds"] = int((d.ic_oos > 0).sum())
            entry["sign_consistent"] = bool(entry["ic_pos_folds"] >= 18)
            em = era_median(d, "capture_q5")
            entry["era_capture_q5"] = {str(k): float(v) for k, v in em.items()}
            entry["eras_clearing_cost"] = int(sum(v >= RT_COST_ATR for v in em.values()))
        else:
            em_auc = era_median(d, "auc_oos")
            entry["auc_median"] = float(d.auc_oos.median())
            entry["era_auc"] = {str(k): float(v) for k, v in em_auc.items()}
            entry["sign_consistent"] = bool(sum(v >= 0.53 for v in em_auc.values()) >= 2)
            ps = p_star["canon" if "canon" in label else "modal"]
            em_td = era_median(d, "topdec_tp_rate")
            entry["era_topdec_tp_rate"] = {str(k): float(v) for k, v in em_td.items()}
            entry["eras_clearing_cost"] = int(sum(v >= ps for v in em_td.values()))
        any_sign |= entry["sign_consistent"]
        any_clear |= entry["eras_clearing_cost"] >= 1
        entry["clears_2plus_eras"] = bool(entry["eras_clearing_cost"] >= 2)
        summary["verdict_configs"][f"{label}|{model}"] = entry

    clears2 = any(v["clears_2plus_eras"] for v in summary["verdict_configs"].values())
    if clears2:
        verdict = "CEILING-CLEARS"
    elif any_sign:
        verdict = "THIN-BUT-REAL"
    elif not any_clear:
        verdict = "NULL"
    else:  # clearance in exactly 1 era without sign-consistency anywhere
        verdict = "THIN-BUT-REAL"
    summary["verdict"] = verdict

    # Widening arm: paired 5y vs 10y OOS IC on qualifying folds (real fits only).
    for label, model in [("fwd4", "ridge"), ("fwd4", "hgb")]:
        b5 = base[(base.label == label) & (base.model == model)].set_index("fold")
        w10 = res[(res.arm == "wide10y") & (res.kind == "real") &
                  (res.label == label) & (res.model == model)].set_index("fold")
        common = sorted(set(b5.index) & set(w10.index))
        d5 = b5.loc[common, "ic_oos"].to_numpy()
        d10 = w10.loc[common, "ic_oos"].to_numpy()
        stat = wilcoxon(d10, d5) if len(common) >= 8 and not np.allclose(d10, d5) \
            else None
        summary["widening"][f"{label}|{model}"] = {
            "n_folds": len(common),
            "ic_median_5y": float(np.median(d5)) if common else None,
            "ic_median_10y": float(np.median(d10)) if common else None,
            "wilcoxon_p": float(stat.pvalue) if stat else None,
            "folds_improved": int((d10 > d5).sum()) if common else None,
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nVERDICT (pinned rules): {verdict}")
    print(f"Total runtime: {time.time() - t_start:,.0f}s")


if __name__ == "__main__":
    main()
