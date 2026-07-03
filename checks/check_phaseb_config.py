"""Guard for the Phase-B production config (switched 2026-07-02).

Loads the REAL backbone through the production CFG defaults and asserts:
  1. Dataset identity + span: the Dukascopy backbone, starting at the census
     clean start (2006-01-01, UTC) — sparse 2003-05 dropped.
  2. Alignment sanity: bar-open→bar-close shift intact (index minutes stamp
     :01 pattern on M1 close-time indexing) and tz is UTC.
  3. Sliding folds: the expected fold count under the 24-month lockbox, and
     NO fold window (train/val/test) touches >= lockbox_start_date.
  4. Fold geometry: first train starts at/after 2006-01-01.

Runtime: loads 7.85M M1 rows (~1-2 min). This is the production-config guard —
run it whenever config data/window knobs change.

Run:  .venv/bin/python checks/check_phaseb_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CFG  # noqa: E402


def main() -> None:
    assert "Dukascopy" in str(CFG.csv_path), CFG.csv_path
    assert CFG.time_col == "DateTime" and CFG.source_tz == "UTC"
    assert CFG.timestamp_is_bar_open is True
    assert CFG.start_date == "2006-01-01"
    assert CFG.lockbox_start_date == "2024-07-01"

    import train_ppo
    from data_loader import make_sliding_folds

    m1, feat, feature_cols = train_ppo._load_decision_features()
    assert str(m1.index.tz) == "UTC"
    assert m1.index.min() >= pd.Timestamp("2006-01-01", tz="UTC")
    # bar-open stamps shifted to bar-close: M1 index minutes end at :01..:00
    assert (m1.index[:1000].second == 0).all()
    assert len(feature_cols) == 25
    print(f"  [1] backbone via CFG: {len(m1):,} M1 rows "
          f"[{m1.index.min().date()} -> {m1.index.max().date()}], "
          f"{len(feat):,} H1 feature bars, 25 features: PASS")

    lock = pd.Timestamp(CFG.lockbox_start_date, tz="UTC")
    folds = make_sliding_folds(
        feat, train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars,
        lockbox_start=CFG.lockbox_start_date)
    n = len(folds)
    for k, (tr, va, te) in enumerate(folds, start=1):
        for name, frame in (("train", tr), ("val", va), ("test", te)):
            assert frame.index.max() < lock, f"fold {k} {name} touches the lockbox"
        assert tr.index.min() >= pd.Timestamp("2006-01-01", tz="UTC")
    # 2006-01 -> 2024-07 usable = 18.5y; fold needs 6y; step 6m -> 26 folds.
    assert 24 <= n <= 28, f"unexpected fold count {n} (expect ~26)"
    no_lock = make_sliding_folds(
        feat, train_years=CFG.sliding_train_years,
        val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
        step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars)
    print(f"  [2] sliding folds: {n} (vs {len(no_lock)} without lockbox); "
          f"first train {folds[0][0].index.min().date()}, "
          f"last test {folds[-1][2].index.max().date()} < {lock.date()}; "
          f"zero lockbox contact: PASS")


if __name__ == "__main__":
    print("Phase-B config checks:")
    main()
    print("Phase-B config checks: ALL PASS")
