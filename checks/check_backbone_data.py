"""Validation gates for the Dukascopy XAUUSD M1 backbone (doc 04 §1.3-1.4).

Gates (1-2, 4 are hard PASS/FAIL; 3 records a verdict; 5 is a census):
  1. Splice vs OANDA overlap — Dukascopy MID (from raw bid+ask chunks) vs the
     OANDA mid at identical UTC instants must agree within spread.
  2. Timezone/alignment — the lag sweep must be minimal at lag 0 by a wide
     margin (a ~1h offset = DST/tz bug).
  3. N6 volume comparability — Spearman of vendor volumes on common bars;
     verdict COMPARABLE (>= 0.7) or REJECT, recorded either way.
  4. Loads + leakage — the backbone loads through load_mt_ohlcv_csv →
     resample H1 → prepare_feature_frame (25 features, 0 NaN/inf) and passes
     leakage_checks.assert_feature_stability_when_future_appended.
  5. Census — per-year bars / days-with-data / longest gap, so the usable
     clean start date can be chosen (early Dukascopy years can be sparse).

Run:  .venv/bin/python checks/check_backbone_data.py [--backbone data/XAUUSD_M1_Bid_Dukascopy_*.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OANDA = ROOT / "data" / "XAU_USD_M1.csv"
RAW = ROOT / "data" / "dukascopy_raw"


def _load_duka_mid_overlap() -> pd.DataFrame:
    """Dukascopy mid = (bid+ask)/2 from the raw 2023+ chunks (splice window)."""
    def load(price: str) -> pd.DataFrame:
        chunks = sorted(RAW.glob(f"xauusd-m1-{price}-202[3-9].csv"))
        if not chunks:
            raise FileNotFoundError(f"no {price} 2023+ chunks in {RAW}")
        df = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return (df.sort_values("dt").drop_duplicates("dt", keep="last")
                  .set_index("dt")[["close", "volume"]])

    bid, ask = load("bid"), load("ask")
    j = bid.join(ask, how="inner", lsuffix="_b", rsuffix="_a")
    j["mid"] = (j["close_b"] + j["close_a"]) / 2.0
    j["spread"] = j["close_a"] - j["close_b"]
    return j


def gate_1_2_3(duka: pd.DataFrame) -> bool:
    oa = pd.read_csv(OANDA)
    oa["dt"] = pd.to_datetime(oa["DateTime"], utc=True)
    oa = oa.set_index("dt")[["Close", "Volume"]]

    j = duka.join(oa, how="inner").dropna(subset=["Close"])
    d = j["mid"] - j["Close"]
    med_spread = float(j["spread"].median())
    med_abs, mean_d = float(d.abs().median()), float(d.mean())
    p99, mx = float(d.abs().quantile(0.99)), float(d.abs().max())

    print(f"\n[1] SPLICE vs OANDA overlap "
          f"({j.index.min().date()} -> {j.index.max().date()}, n={len(j):,})")
    print(f"    median|mid diff| = {med_abs:.4f}   mean = {mean_d:+.4f}   "
          f"p99 = {p99:.4f}   max = {mx:.4f}   (median spread {med_spread:.3f})")
    g1 = med_abs <= med_spread and abs(mean_d) <= 0.5 * med_spread
    print(f"    -> {'PASS' if g1 else 'FAIL'} (need median|d| <= spread, |mean| <= spread/2)")

    print("\n[2] TIMEZONE/ALIGNMENT (lag sweep, minutes)")
    lag_med = {}
    for lag in (-120, -60, 0, 60, 120):
        sh = duka.copy()
        sh.index = sh.index + pd.Timedelta(minutes=lag)
        jj = sh.join(oa[["Close"]], how="inner").dropna(subset=["Close"])
        lag_med[lag] = float((jj["mid"] - jj["Close"]).abs().median())
        print(f"    lag {lag:+4d}: median|d| = {lag_med[lag]:.4f}")
    others = min(v for k, v in lag_med.items() if k != 0)
    g2 = lag_med[0] < others / 5.0
    print(f"    -> {'PASS' if g2 else 'FAIL'} (lag 0 minimal by >=5x)")

    v = j[["volume_b", "Volume"]].replace(0, np.nan).dropna()
    rho = float(v["volume_b"].corr(v["Volume"], method="spearman"))
    verdict = "COMPARABLE" if rho >= 0.7 else "REJECT"
    print(f"\n[3] N6 VOLUME COMPARABILITY: Spearman={rho:.3f} (n={len(v):,}) -> {verdict}")
    if verdict == "REJECT":
        print("    VERDICT: volume features REJECTED — vendor tick volumes do not "
              "reconcile; nothing downstream may use Volume as a feature (doc 05 N6).")
    else:
        print("    VERDICT: volumes rank-comparable across vendors (units differ; "
              "any volume feature must be scale-free, doc 05 N6/P-batch rules).")
    return g1 and g2


def gate_4(backbone: Path) -> bool:
    from config import CFG
    from data_loader import load_mt_ohlcv_csv, resample_ohlcv
    from features import prepare_feature_frame
    from leakage_checks import assert_feature_stability_when_future_appended

    print(f"\n[4] LOADS + LEAKAGE ({backbone.name})")
    m1 = load_mt_ohlcv_csv(backbone, time_col="DateTime", source_tz="UTC",
                           timestamp_is_bar_open=True, bar_duration="1min")
    h1 = resample_ohlcv(m1, "1h")
    feat, cols = prepare_feature_frame(h1, warmup_bars=CFG.warmup_bars,
                                       atr_period=CFG.atr_period,
                                       rsi_period=CFG.rsi_period)
    n_nan = int(feat[cols].isna().sum().sum())
    n_inf = int(np.isinf(feat[cols].to_numpy(dtype=float)).sum())
    print(f"    M1 rows {len(m1):,} [{m1.index.min()} -> {m1.index.max()}]")
    print(f"    H1 feature bars {len(feat):,}; features {len(cols)} (expect 25); "
          f"NaN/inf {n_nan}/{n_inf} (expect 0/0)")
    ok_feat = len(cols) == 25 and n_nan == 0 and n_inf == 0
    leak_ok = assert_feature_stability_when_future_appended(
        h1, cols, cut_index=int(len(h1) * 0.6),
        atr_period=CFG.atr_period, rsi_period=CFG.rsi_period)
    print(f"    leakage guardrail: {'PASS' if leak_ok else 'FAIL'}")
    print(f"    -> {'PASS' if (ok_feat and leak_ok) else 'FAIL'}")
    return ok_feat and leak_ok


def gate_5_census(backbone: Path) -> None:
    df = pd.read_csv(backbone, usecols=["DateTime"])
    dt = pd.to_datetime(df["DateTime"], utc=True)
    print("\n[5] CENSUS (pick the usable clean start from this)")
    print(f"    span: {dt.min()} -> {dt.max()}  "
          f"({(dt.max() - dt.min()).days / 365.25:.1f} years, {len(dt):,} bars)")
    by_year = dt.groupby(dt.dt.year)
    gaps = dt.diff()
    print("    year  bars      days  longest-gap")
    for y, s in by_year:
        g = gaps[dt.dt.year == y].max()
        print(f"    {y}  {len(s):>8,}  {s.dt.date.nunique():>4}  "
              f"{'' if pd.isna(g) else str(g)}")
    top = gaps.nlargest(3)
    print("    3 longest gaps overall (weekends ~2d2h are normal):")
    for i in top.index:
        print(f"      {gaps[i]}   ending {dt[i]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None)
    args = ap.parse_args()
    backbone = (Path(args.backbone) if args.backbone else
                sorted((ROOT / "data").glob("XAUUSD_M1_Bid_Dukascopy_*.csv"))[-1])

    print("Backbone validation gates (doc 04 §1.3-1.4):")
    duka = _load_duka_mid_overlap()
    g123 = gate_1_2_3(duka)
    g4 = gate_4(backbone)
    gate_5_census(backbone)
    if not (g123 and g4):
        print("\nBACKBONE VALIDATION: FAILED — do not stitch/use until resolved.")
        raise SystemExit(1)
    print("\nBACKBONE VALIDATION: ALL HARD GATES PASS")
