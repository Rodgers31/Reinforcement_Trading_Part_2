"""Convert raw dukascopy-node CSV chunks into one LEAN-convention M1 CSV.

Input : data/dukascopy_raw/xauusd-m1-<price>-<label>.csv  (epoch-ms UTC
        bar-open `timestamp`, open/high/low/close, volume-in-units) — the
        per-year chunks written by the resumable pull.
Output: data/XAUUSD_M1_<Price>_Dukascopy_<start>_<end>.csv with the existing
        LEAN convention (DateTime,Open,High,Low,Close,Volume; UTC; bar-open),
        so it drops into the loader with time_col="DateTime", source_tz="UTC",
        timestamp_is_bar_open=True (doc 04 §1.1).

The conversion is faithful: parse, concat, sort, de-duplicate timestamps
(keep last). OHLC sanity filtering stays where it already lives —
data_loader.load_mt_ohlcv_csv.

Run:  .venv/bin/python convert_dukascopy_to_lean.py [--price bid]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def convert(price: str = "bid", raw_dir: Path = ROOT / "data" / "dukascopy_raw",
            out_dir: Path = ROOT / "data") -> Path:
    chunks = sorted(raw_dir.glob(f"xauusd-m1-{price}-*.csv"))
    if not chunks:
        raise FileNotFoundError(f"no xauusd-m1-{price}-*.csv chunks in {raw_dir}")

    frames = []
    for c in chunks:
        df = pd.read_csv(c)
        need = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{c.name}: missing columns {missing}")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    raw["DateTime"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = (raw.sort_values("DateTime")
              .drop_duplicates(subset="DateTime", keep="last"))

    out = pd.DataFrame({
        "DateTime": raw["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": raw["open"], "High": raw["high"],
        "Low": raw["low"], "Close": raw["close"],
        "Volume": raw["volume"],
    })

    start = raw["DateTime"].iloc[0].strftime("%Y.%m.%d")
    end = raw["DateTime"].iloc[-1].strftime("%Y.%m.%d")
    out_path = out_dir / f"XAUUSD_M1_{price.capitalize()}_Dukascopy_{start}_{end}.csv"
    out.to_csv(out_path, index=False)

    print(f"chunks    : {len(chunks)}  ({chunks[0].name} … {chunks[-1].name})")
    print(f"rows      : {len(out):,}  (duplicates dropped: {sum(len(f) for f in frames) - len(out):,})")
    print(f"range     : {raw['DateTime'].iloc[0]} -> {raw['DateTime'].iloc[-1]}")
    print(f"span      : {(raw['DateTime'].iloc[-1] - raw['DateTime'].iloc[0]).days / 365.25:.1f} years")
    print(f"output    : {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", default="bid", choices=["bid", "ask"])
    args = ap.parse_args()
    convert(price=args.price)
