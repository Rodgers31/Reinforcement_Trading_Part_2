"""Task 27 Track C — freeze the shadow model (pre-lockbox 10y V1 refit).

Trains the frozen forward-shadow scorer: Ridge(alpha=1.0) on train-standardized
features, fwd4 label, train window = the 10y ending at the LAST PRE-LOCKBOX bar
(2014-07-01 -> 2024-06-30). Deliberately NOT a trailing-to-present refit: that
would train on lockbox-era bars (seal preserved; refit cadence suspended per
the Task-27 pin). Writes enhancements/12_shadow/frozen_model.json with scaler,
coefficients, q80 threshold, window metadata, and a sha256 params hash.

Run once: .venv/bin/python enhancements/12_shadow/freeze_model.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import CFG  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402

OUT = Path(__file__).resolve().parent
K, Q = 4, 0.8


def main() -> None:
    print("Loading data (anchor-identical path)…")
    _, feat, feature_cols = _load_decision_features()
    assert len(feature_cols) == 25
    cut = pd.Timestamp(CFG.lockbox_start_date).tz_localize(feat.index.tz)
    t0 = (cut - pd.DateOffset(months=120))
    flb = feat.loc[feat.index < cut]
    train = flb.loc[flb.index >= t0]
    print(f"train window: {train.index.min()} -> {train.index.max()} ({len(train):,} bars)")

    close = train["Close"].to_numpy(float)
    atr = train["atr"].to_numpy(float)
    n = len(train)
    y = np.full(n, np.nan)
    y[: n - K] = (close[K:] - close[:-K]) / atr[: n - K]
    m = ~np.isnan(y)
    X = train[feature_cols].to_numpy(float)[m]
    sc = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(sc.transform(X), y[m])
    thr = float(np.quantile(np.abs(model.predict(sc.transform(X))), Q))

    params = {
        "frozen_at": "2026-07-08",
        "rule": "V1: fwd4-ridge, |score|>=q80 -> side=sign(score), k=4 time exit (paper)",
        "train_window": [str(train.index.min()), str(train.index.max())],
        "n_train": int(m.sum()),
        "k": K, "q": Q,
        "feature_cols": feature_cols,
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "thr_q80": thr,
        "refit_policy": "SUSPENDED — any refit would train on lockbox bars; needs a new pin",
    }
    blob = json.dumps(params, sort_keys=True).encode()
    params["model_hash"] = hashlib.sha256(blob).hexdigest()[:16]
    (OUT / "frozen_model.json").write_text(json.dumps(params, indent=2))
    print(f"frozen_model.json written | hash {params['model_hash']} | thr_q80 {thr:.6f} "
          f"| n_train {params['n_train']:,}")


if __name__ == "__main__":
    main()
