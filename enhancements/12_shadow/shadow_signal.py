"""Task 27 Track C — forward-shadow signal logger (paper only; no broker).

Weekly cadence (manual or user-scheduled). Loads the FROZEN model (never
refits), obtains the latest XAUUSD M1 bars — fresh Dukascopy pull of the last
~35 days when the network allows, else the repo backbone file's tail (the
`data_source` column records which) — rebuilds the 25 features, scores the
last COMPLETED H1 bar, and appends one row to the append-only shadow_log.csv:

  utc_bar_close, close, atr, score, thr, side, threshold_pass, data_source, model_hash

Duplicate bar timestamps are refused (idempotent re-runs). No orders, no
execution simulation — a timestamped forward record of what the frozen rule
would have said.

Run: .venv/bin/python enhancements/12_shadow/shadow_signal.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from config import CFG  # noqa: E402
from data_loader import load_mt_ohlcv_csv, resample_ohlcv  # noqa: E402
from features import prepare_feature_frame  # noqa: E402

OUT = Path(__file__).resolve().parent
LOG = OUT / "shadow_log.csv"
PULL_DAYS = 35


def fresh_m1() -> pd.DataFrame | None:
    """Try a fresh Dukascopy pull of the last PULL_DAYS; None on failure."""
    today = datetime.now(timezone.utc).date()
    frm = (today - timedelta(days=PULL_DAYS)).isoformat()
    to = (today + timedelta(days=1)).isoformat()
    tmp = tempfile.mkdtemp(prefix="shadow_pull_")
    try:
        subprocess.run(
            ["npx", "--yes", "dukascopy-node", "-i", "xauusd", "-from", frm,
             "-to", to, "-t", "m1", "-p", "bid", "-f", "csv",
             "-bs", "3", "-bp", "2000", "-r", "3", "-rp", "3000", "-dir", tmp],
            capture_output=True, timeout=600)
        files = sorted(Path(tmp).glob("xauusd-m1-bid-*.csv"))
        if not files:
            return None
        df = pd.read_csv(files[-1])
        if len(df) < 1000:
            return None
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
        out = pd.DataFrame({"Open": df["open"], "High": df["high"], "Low": df["low"],
                            "Close": df["close"], "Volume": df["volume"]})
        out.index = idx  # bar-CLOSE stamps, matching the loader convention
        return out.sort_index()
    except Exception:
        return None


def backbone_tail() -> pd.DataFrame:
    m1 = load_mt_ohlcv_csv(
        CFG.csv_path, time_col=CFG.time_col, source_tz=CFG.source_tz,
        timestamp_is_bar_open=CFG.timestamp_is_bar_open,
        bar_duration=CFG.pandas_execution_tf,
        start_date=None, end_date=None, max_days_for_demo=None)
    return m1.iloc[-PULL_DAYS * 24 * 60:]


def main() -> None:
    frozen = json.loads((OUT / "frozen_model.json").read_text())
    m1 = fresh_m1()
    source = "dukascopy_fresh"
    if m1 is None:
        m1 = backbone_tail()
        source = "backbone_file_tail"
    h1 = resample_ohlcv(m1, CFG.pandas_tf)
    # drop the final H1 bar if the M1 stream doesn't reach its close time
    if len(h1) and m1.index.max() < h1.index.max():
        h1 = h1.iloc[:-1]
    feat, cols = prepare_feature_frame(h1, warmup_bars=CFG.warmup_bars,
                                       atr_period=CFG.atr_period,
                                       rsi_period=CFG.rsi_period)
    assert cols == frozen["feature_cols"], "feature set drifted vs frozen model"
    row = feat.iloc[-1]
    x = row[cols].to_numpy(float)
    z = (x - np.array(frozen["scaler_mean"])) / np.array(frozen["scaler_scale"])
    score = float(np.dot(z, np.array(frozen["coef"])) + frozen["intercept"])
    thr = frozen["thr_q80"]
    passed = abs(score) >= thr
    side = int(np.sign(score)) if passed else 0

    ts = str(feat.index[-1])
    if LOG.exists():
        prev = pd.read_csv(LOG)
        if len(prev) and str(prev.iloc[-1]["utc_bar_close"]) == ts:
            print(f"[skip] bar {ts} already logged — append-only log unchanged")
            return
    new = not LOG.exists()
    with open(LOG, "a") as f:
        if new:
            f.write("utc_bar_close,close,atr,score,thr,side,threshold_pass,"
                    "data_source,model_hash\n")
        f.write(f"{ts},{row['Close']:.3f},{row['atr']:.4f},{score:.6f},{thr:.6f},"
                f"{side},{passed},{source},{frozen['model_hash']}\n")
    print(f"[logged] {ts} close {row['Close']:.2f} score {score:+.4f} thr {thr:.4f} "
          f"side {side} pass {passed} source {source}")


if __name__ == "__main__":
    main()
