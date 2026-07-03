"""Repro/check for the lockbox-carve mechanism (doc 04 Phase A).

The lockbox is a tail period excluded from the ENTIRE sliding walk-forward
sweep: no fold's train/val/test window may contain any bar at/after
lockbox_start. Asserts, on synthetic 8y hourly data (naive AND tz-aware):
  1. with a lockbox: every fold frame ends strictly before the cut, and the
     folds are identical to running on the pre-truncated frame (equivalence);
  2. without a lockbox: folds do reach past the cut date (the carve matters);
  3. lockbox_start=None is a no-op (default behavior preserved).

Run:  .venv/bin/python checks/check_lockbox_carve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import make_sliding_folds  # noqa: E402

LOCKBOX = "2025-01-01"


def _frame(tz=None) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", "2026-01-01", freq="1h", tz=tz)
    return pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)


def _fold_ids(folds):
    return [(len(tr), len(va), len(te),
             str(tr.index.min()), str(te.index.max())) for tr, va, te in folds]


def check(tz) -> None:
    df = _frame(tz)
    cut = pd.Timestamp(LOCKBOX, tz=tz)

    plain = make_sliding_folds(df)
    carved = make_sliding_folds(df, lockbox_start=LOCKBOX)

    # (2) without the carve, the sweep reaches past the cut (the carve matters).
    assert any(te.index.max() >= cut for _, _, te in plain), \
        "synthetic frame too short — no fold reaches the lockbox period"

    # (1) with the carve, NO bar in ANY window is at/after the cut...
    assert carved, "carve produced zero folds"
    for k, (tr, va, te) in enumerate(carved, start=1):
        for name, frame in (("train", tr), ("val", va), ("test", te)):
            assert frame.index.max() < cut, \
                f"fold {k} {name} touches the lockbox ({frame.index.max()} >= {cut})"
    # ...and the result is exactly what running on a pre-truncated frame gives.
    assert _fold_ids(carved) == _fold_ids(make_sliding_folds(df.loc[df.index < cut])), \
        "carved folds differ from folds on the pre-truncated frame"
    assert len(carved) < len(plain), "carve removed no folds"

    # (3) None is a no-op.
    assert _fold_ids(make_sliding_folds(df, lockbox_start=None)) == _fold_ids(plain)

    print(f"  tz={tz or 'naive'}: {len(plain)} folds -> {len(carved)} carved, "
          f"all windows < {LOCKBOX}, equivalence + no-op hold: PASS")


if __name__ == "__main__":
    print("Lockbox-carve checks:")
    check(tz=None)
    check(tz="UTC")   # real data is tz-aware; naive config date must localize
    print("Lockbox-carve checks: ALL PASS")
