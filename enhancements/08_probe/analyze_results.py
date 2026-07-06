"""Task 22 probe — post-run analysis (research-only, no fits).

Reads results_per_fold.csv / results_univariate.csv / fold_windows.csv and
prints the tables for enhancements/08. Also reloads the H1 frame once to
compute per-era sigma(fwd4) for the IC_req translation (a derived context
number, not a verdict input — verdict came from pinned capture/p* rules).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:0.4f}")

res = pd.read_csv(OUT / "results_per_fold.csv")
uni = pd.read_csv(OUT / "results_univariate.csv")
win = pd.read_csv(OUT / "fold_windows.csv")

base = res[(res.arm == "base5y") & (res.kind == "real")]
shuf = res[res.kind == "shuffle"]
wide = res[(res.arm == "wide10y") & (res.kind == "real")]
RT = 0.0683

print("=" * 100)
print("1) INTEGRITY AUDIT (BLAS-warning check): NaN counts + selection fractions by model")
for col, sub in [("ic_oos", base[base.label.str.startswith("fwd")]),
                 ("auc_oos", base[base.label.str.startswith("br_")])]:
    print(f"  {col}: NaN by model:", sub.groupby("model")[col].apply(lambda s: int(s.isna().sum())).to_dict())
print("  q5_frac by model:", base[base.label.str.startswith("fwd")].groupby("model")["q5_frac"].median().round(3).to_dict())
print("  topdec_frac by model:", base[base.label.str.startswith("br_")].groupby("model")["topdec_frac"].median().round(3).to_dict())
print("  ridge |ic_oos| max:", float(base[(base.model == 'ridge')]["ic_oos"].abs().max()))

print("\n2) SHUFFLE NOISE BAND (per-fold OOS metric under permuted train labels)")
sic = shuf[shuf.label.str.startswith("fwd")]["ic_oos"]
sauc = shuf[shuf.label.str.startswith("br_")]["auc_oos"]
print(f"  shuffle IC:  median {sic.median():+.4f}  p5 {sic.quantile(.05):+.4f}  p95 {sic.quantile(.95):+.4f}  (n={len(sic)})")
print(f"  shuffle AUC: median {sauc.median():.4f}  p5 {sauc.quantile(.05):.4f}  p95 {sauc.quantile(.95):.4f}  (n={len(sauc)})")

print("\n3) ERA CALENDAR MAP")
win["test_start"] = pd.to_datetime(win.test_start)
win["test_end"] = pd.to_datetime(win.test_end)
for e, g in win.groupby("era"):
    print(f"  E{e}: folds {g.fold.min():>2}-{g.fold.max():>2}  tests {g.test_start.min().date()} -> {g.test_end.max().date()}")

print("\n4) REGRESSION — era-median OOS IC and capture (all k, both models), cost bar 0.0683")
reg = base[base.label.str.startswith("fwd")]
for metric in ["ic_oos", "capture_q5", "capture_all"]:
    t = reg.pivot_table(index=["label", "model"], columns="era", values=metric, aggfunc="median")
    t["ALL"] = reg.groupby(["label", "model"])[metric].median()
    print(f"  --- {metric} ---")
    print(t.round(4).to_string())

print("\n5) SUPERVISED TRANSFER GAP — train IC vs OOS IC (fwd4)")
f4 = reg[reg.label == "fwd4"]
g = f4.groupby("model")[["ic_train", "ic_oos"]].median()
g["gap"] = g.ic_train - g.ic_oos
print(g.round(4).to_string())
print("  per-era OOS/train ratio (hgb):")
h4 = f4[f4.model == "hgb"]
t = h4.groupby("era")[["ic_train", "ic_oos"]].median()
t["ratio"] = t.ic_oos / t.ic_train
print(t.round(4).to_string())

print("\n6) CLASSIFICATION — era-median AUC and top-decile TP rate (canon p*=0.5342, modal p*=0.2585)")
cls = base[base.label.str.startswith("br_")]
for metric in ["auc_oos", "topdec_tp_rate", "base_rate_test"]:
    t = cls.pivot_table(index=["label", "model"], columns="era", values=metric, aggfunc="median")
    t["ALL"] = cls.groupby(["label", "model"])[metric].median()
    print(f"  --- {metric} ---")
    print(t.round(4).to_string())

print("\n7) PER-FOLD DETAIL — fwd4 hgb + ridge (IC, capture_q5, clears?)")
d = f4.pivot_table(index=["fold", "era"], columns="model", values=["ic_oos", "capture_q5"])
d.columns = [f"{a}_{b}" for a, b in d.columns]
d["clears_hgb"] = d.capture_q5_hgb >= RT
d["clears_ridge"] = d.capture_q5_ridge >= RT
d = d.reset_index().merge(win[["fold", "test_start"]], on="fold")
d["test_start"] = d.test_start.dt.date
print(d.round(4).to_string(index=False))
print(f"  folds clearing cost: hgb {int(d.clears_hgb.sum())}/25, ridge {int(d.clears_ridge.sum())}/25")

print("\n8) UNIVARIATE SCREEN (k=4): top features by |median OOS IC| across 25 folds; E4-only medians")
u = uni.groupby("feature")["ic_oos"].median().sort_values(key=np.abs, ascending=False)
u4 = uni[uni.era == 4].groupby("feature")["ic_oos"].median()
top = u.head(10).to_frame("ic_all_folds")
top["ic_E4_only"] = u4.reindex(top.index)
print(top.round(4).to_string())

print("\n9) WIDENING ARM 5y->10y — per-era median delta-IC (10y minus 5y), fwd4")
for model in ["ridge", "hgb"]:
    b5 = f4[f4.model == model].set_index("fold")["ic_oos"]
    w10 = wide[(wide.label == "fwd4") & (wide.model == model)].set_index("fold")["ic_oos"]
    common = sorted(set(b5.index) & set(w10.index))
    dd = pd.DataFrame({"fold": common,
                       "era": [win.set_index('fold').era[f] for f in common],
                       "d": [w10[f] - b5[f] for f in common]})
    t = dd.groupby("era")["d"].agg(["median", "count", lambda s: int((s > 0).sum())])
    t.columns = ["median_dIC", "n", "improved"]
    print(f"  --- {model} ---")
    print(t.round(4).to_string())

print("\n10) SIGMA(fwd4) PER ERA -> IC_req = 0.0683 / (0.7979 * sigma)  [context only]")
from config import CFG  # noqa: E402
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402
_, feat, feature_cols = _load_decision_features()
folds = make_sliding_folds(feat, train_years=CFG.sliding_train_years,
                           val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
                           step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars,
                           lockbox_start=CFG.lockbox_start_date)
rows = []
for i, (tr, va, te) in enumerate(folds, start=1):
    close, a = te["Close"].to_numpy(float), te["atr"].to_numpy(float)
    k = 4
    y = (close[k:] - close[:-k]) / a[:-k]
    rows.append({"fold": i, "era": win.set_index('fold').era[i], "sigma4": float(np.std(y))})
sg = pd.DataFrame(rows).groupby("era")["sigma4"].median().to_frame("sigma4")
sg["IC_req"] = RT / (0.7979 * sg.sigma4)
print(sg.round(4).to_string())
print("\nDONE")
