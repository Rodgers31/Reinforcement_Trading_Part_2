"""Task 27 Track B step 1 — convert XAGUSD chunks, density census, cost bar.

Mirrors the gold 0b methodology (Task-27 pin):
  - convert per-year chunks -> LEAN CSVs (bid backbone + ask 2023+)
  - census: per-year M1 bid rows; CLEAN START = first year with >=300k rows
    and all later years >=300k (gold's dense band is 333-365k)
  - cost: spread_atr_frac = median(ask-bid on common M1 closes) /
    median(H1 ATR14) over 2023-01 -> data end; slippage_atr_frac = 0.0030
    (methodology reuse, pinned); RT = spread + 2*slip

Writes enhancements/12_silver/silver_census_cost.json.
Run: .venv/bin/python enhancements/12_silver/silver_census_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from convert_dukascopy_to_lean import convert  # noqa: E402
from data_loader import load_mt_ohlcv_csv, resample_ohlcv  # noqa: E402
from features import atr as atr_fn  # noqa: E402

OUT = Path(__file__).resolve().parent
DENSE_ROWS = 300_000
SLIPPAGE_ATR_FRAC = 0.0030
CENSUS_COST_START = "2023-01-01"


def main() -> None:
    print("converting chunks…")
    bid_path = convert(price="bid", symbol="xagusd")
    ask_path = convert(price="ask", symbol="xagusd")

    bid = load_mt_ohlcv_csv(bid_path, time_col="DateTime", source_tz="UTC",
                            timestamp_is_bar_open=True, bar_duration="1min")
    per_year = bid.groupby(bid.index.year).size()
    print("\nper-year M1 bid rows:")
    for y, n in per_year.items():
        print(f"  {y}: {n:,}")
    clean_start = None
    years = sorted(per_year.index)
    for y in years:
        if all(per_year.get(z, 0) >= DENSE_ROWS for z in years if y <= z < 2026):
            clean_start = y
            break
    assert clean_start is not None, "no dense start year found"
    print(f"CLEAN START: {clean_start}-01-01 (>= {DENSE_ROWS:,} rows/yr thereafter)")

    ask = load_mt_ohlcv_csv(ask_path, time_col="DateTime", source_tz="UTC",
                            timestamp_is_bar_open=True, bar_duration="1min")
    b = bid.loc[bid.index >= CENSUS_COST_START, "Close"]
    a = ask.loc[ask.index >= CENSUS_COST_START, "Close"]
    common = b.index.intersection(a.index)
    spread = (a.loc[common] - b.loc[common])
    med_spread = float(spread.median())
    h1 = resample_ohlcv(bid.loc[bid.index >= CENSUS_COST_START], "1h")
    med_atr = float(atr_fn(h1, 14).median())
    spread_frac = med_spread / med_atr
    rt = spread_frac + 2 * SLIPPAGE_ATR_FRAC

    out = {
        "bid_path": str(bid_path), "ask_path": str(ask_path),
        "per_year_rows": {str(y): int(n) for y, n in per_year.items()},
        "clean_start": f"{clean_start}-01-01",
        "cost_window": [CENSUS_COST_START, str(bid.index.max().date())],
        "n_common_spread_minutes": int(len(common)),
        "median_spread": med_spread,
        "median_h1_atr14": med_atr,
        "spread_atr_frac": round(spread_frac, 4),
        "slippage_atr_frac": SLIPPAGE_ATR_FRAC,
        "rt_cost_atr": round(rt, 4),
        "p_star_canon": round((1.0 + rt / 1.0) / 2.0, 4),
        "p_star_modal": round((1.0 + rt / 2.0) / 4.0, 4),
    }
    (OUT / "silver_census_cost.json").write_text(json.dumps(out, indent=2))
    print(f"\nspread median {med_spread:.4f} | H1 ATR14 median {med_atr:.4f} | "
          f"spread_atr_frac {spread_frac:.4f} | RT bar {rt:.4f} "
          f"(gold: 0.0623 / 0.0683)")
    print("silver_census_cost.json written")


if __name__ == "__main__":
    main()
