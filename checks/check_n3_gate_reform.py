"""Repro/check for the ratified N3 gate re-form (doc 05 N3, pinned 2026-07-01).

Fraction/quantile gate: breadth >=70% of folds (return>0 AND PF>1), floor
10th-percentile PF >= 0.90, third leg mean TRADE-based Sharpe > 0. The gate's
meaning must now be invariant to fold count:
  - n=5 : ceil(0.70*5)=4 reproduces the OLD 4-of-5 breadth exactly.
  - n=34: breadth needs 24 (was: 4 = trivial); the floor tolerates ~3 bad
    folds (was: one bad fold vetoed all 34).
Also asserts the Ulcer secondary metric (non-gating) arithmetic.

DISCIPLINE (recorded with the pin): a sound gate rejecting the baseline is a
RESULT, not a trigger to loosen; re-pin only for mechanical mis-specification,
never to make a result pass.

Run:  .venv/bin/python checks/check_n3_gate_reform.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_ppo import _passes_consistency_gate  # noqa: E402
from evaluate import ulcer_index  # noqa: E402


def _summary(rets, pfs, sharpes) -> pd.DataFrame:
    return pd.DataFrame({
        "val_return_pct": rets,
        "val_profit_factor": pfs,
        "val_sharpe_trade": sharpes,
    })


def _gate(df) -> bool:
    passed, detail = _passes_consistency_gate(df)
    return passed


def n5_checks() -> None:
    # 4 of 5 good (old 4-of-5 semantics) → breadth OK; healthy floor/sharpe → PASS.
    ok = _summary([5, 4, 3, 2, -1], [1.3, 1.25, 1.2, 1.15, 0.98], [0.5] * 5)
    assert _gate(ok) is True

    # Only 3 of 5 good → breadth FAIL (matches the old gate's 4-of-5).
    assert _gate(_summary([5, 4, 3, -1, -2], [1.3, 1.2, 1.1, 0.95, 0.9],
                          [0.5] * 5)) is False

    # Floor at n=5: one catastrophic fold (PF 0.5) → q10 = 0.68 < 0.90 → FAIL.
    assert _gate(_summary([5, 4, 3, 2, -1], [1.3, 1.25, 1.2, 1.15, 0.5],
                          [0.5] * 5)) is False

    # Documented divergence from the OLD floor: worst=0.85 (old gate: FAIL at
    # 0.85<0.90) now blends q10 = 0.85+0.4*(1.15-0.85) = 0.97 → floor PASSES.
    # The quantile form was ratified knowing this smoothing at tiny n.
    div = _summary([5, 4, 3, 2, -1], [1.3, 1.25, 1.2, 1.15, 0.85], [0.5] * 5)
    assert _gate(div) is True

    # Negative mean trade-Sharpe → third leg FAIL.
    assert _gate(_summary([5, 4, 3, 2, -1], [1.3, 1.25, 1.2, 1.15, 0.98],
                          [-0.1] * 5)) is False
    print("  [1] n=5: breadth reproduces old 4-of-5; floor + sharpe legs behave: PASS")


def n34_checks() -> None:
    rng = np.random.default_rng(0)

    def mk(n_good: int, bad_pfs: list[float]) -> pd.DataFrame:
        n = 34
        n_bad = n - n_good
        rets = [1.0] * n_good + [-1.0] * n_bad
        pfs = [1.2] * n_good + (bad_pfs + [0.95] * (n_bad - len(bad_pfs)))
        return _summary(rets, pfs, [0.3] * n)

    # 24/34 good (>= ceil(0.7*34)=24) → PASS (floor: q10 of PFs stays >= 0.90).
    assert _gate(mk(24, bad_pfs=[0.95])) is True
    # 23/34 good → breadth FAIL. (Old gate: 23 >> 4 → would have trivially passed.)
    assert _gate(mk(23, bad_pfs=[0.95])) is False

    # Floor at n=34: 3 catastrophic folds tolerated, 4 rejected.
    good31 = [1.3] * 31
    three_bad = _summary([1.0] * 31 + [-1.0] * 3, good31 + [0.5] * 3, [0.3] * 34)
    four_bad = _summary([1.0] * 30 + [-1.0] * 4, [1.3] * 30 + [0.5] * 4, [0.3] * 34)
    assert _gate(three_bad) is True   # q10 ≈ 1.3 (3.3rd order stat) >= 0.90
    assert _gate(four_bad) is False   # q10 ≈ 0.74 < 0.90
    # (Old gate: even ONE 0.5-PF fold among 34 vetoed the whole run.)

    # NaN PFs (no-trade folds) are skipped by the quantile, not treated as 0.
    nan_ok = _summary([1.0] * 24 + [-1.0] * 10,
                      [1.2] * 24 + [np.nan] * 10, [0.3] * 34)
    assert _gate(nan_ok) is True
    print("  [2] n=34: breadth 24-not-4; floor tolerates 3 bad, rejects 4; NaN-safe: PASS")


def ulcer_check() -> None:
    flat = pd.Series([100.0, 100.0, 100.0])
    assert abs(ulcer_index(flat)) < 1e-12
    # equity [100, 90, 100]: dd% = [0, -10, 0] → RMS = sqrt(100/3) = 5.7735%.
    dip = pd.Series([100.0, 90.0, 100.0])
    assert abs(ulcer_index(dip) - np.sqrt(100.0 / 3.0)) < 1e-9
    assert np.isnan(ulcer_index(pd.Series(dtype=float)))
    print("  [3] Ulcer secondary (non-gating) arithmetic: PASS")


if __name__ == "__main__":
    print("N3 gate-reform checks:")
    n5_checks()
    n34_checks()
    ulcer_check()
    print("N3 gate-reform checks: ALL PASS")
