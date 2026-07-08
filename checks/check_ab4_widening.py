"""Pre-launch checks for A/B #4 (Task 24): the 10y widened grid and its guards.

Asserts, before any training job starts:
  1. ANCHOR PARITY — with SLIDING_TRAIN_YEARS unset, make_sliding_folds yields
     the 25 anchor folds with windows identical to the Task-22
     fold_windows.csv record.
  2. GRID EQUIVALENCE — with train_years=10 the grid yields exactly 15 folds
     whose val/test windows are index-identical to the 5y grid's folds 11-25,
     and whose train windows end at the same bar (only the start moves back
     ~5y, matching the Task-22/23 widened-boundary formula).
  3. CONFIG OVERRIDE — SLIDING_TRAIN_YEARS=10 reaches CFG in a fresh
     interpreter; invalid values raise.
  4. FAIL-FAST GUARD — run_baseline refuses to launch a non-anchor
     sliding_train_years without --candidate (exits nonzero before creating a
     run dir).

Run: .venv/bin/python checks/check_ab4_widening.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import CFG  # noqa: E402
from data_loader import make_sliding_folds  # noqa: E402
from train_ppo import _load_decision_features  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")


def main() -> None:
    assert CFG.sliding_train_years == 5.0, \
        "run this check with SLIDING_TRAIN_YEARS unset (anchor config)"

    print("loading data once (anchor path)…")
    _, feat, _ = _load_decision_features()
    kw = dict(val_months=CFG.sliding_val_months, test_months=CFG.sliding_test_months,
              step_months=CFG.sliding_step_months, embargo_bars=CFG.split_embargo_bars,
              lockbox_start=CFG.lockbox_start_date)

    # 1) anchor parity vs the Task-22 record
    f5 = make_sliding_folds(feat, train_years=5.0, **kw)
    assert len(f5) == 25, f"anchor grid changed: {len(f5)} folds"
    rec = pd.read_csv(REPO / "enhancements" / "08_probe" / "fold_windows.csv")
    for i, (tr, va, te) in enumerate(f5, start=1):
        row = rec[rec.fold == i].iloc[0]
        assert str(tr.index.min()) == row.train_start and str(tr.index.max()) == row.train_end, \
            f"fold {i} train window drifted vs Task-22 record"
        assert str(te.index.min()) == row.test_start and str(te.index.max()) == row.test_end, \
            f"fold {i} test window drifted vs Task-22 record"
    print("[OK] 1. anchor parity: 25 folds byte-identical to the Task-22 fold_windows record")

    # 2) 10y grid equivalence: fold j == fold j+10 (val/test), train end shared
    f10 = make_sliding_folds(feat, train_years=10.0, **kw)
    assert len(f10) == 15, f"10y grid: expected 15 folds, got {len(f10)}"
    for j, (tr10, va10, te10) in enumerate(f10, start=1):
        tr5, va5, te5 = f5[j + 10 - 1]
        assert va10.index.equals(va5.index), f"10y fold {j}: val != 5y fold {j+10} val"
        assert te10.index.equals(te5.index), f"10y fold {j}: test != 5y fold {j+10} test"
        assert tr10.index.max() == tr5.index.max(), \
            f"10y fold {j}: train end != 5y fold {j+10} train end"
        assert tr10.index.min() < tr5.index.min(), f"10y fold {j}: train did not widen"
        span_days = (tr10.index.max() - tr10.index.min()).days
        assert 3600 <= span_days <= 3680, f"10y fold {j}: train span {span_days}d not ~10y"
    print("[OK] 2. grid equivalence: 15 folds; val/test == 5y folds 11-25; train widened to ~10y")

    # 3) config override in a fresh interpreter
    env = {**os.environ, "SLIDING_TRAIN_YEARS": "10"}
    out = subprocess.run(
        [PY, "-c", "from config import CFG; print(CFG.sliding_train_years)"],
        env=env, capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0 and out.stdout.strip() == "10.0", \
        f"override failed: {out.stdout!r} {out.stderr[-200:]!r}"
    bad = subprocess.run([PY, "-c", "import config"],
                         env={**os.environ, "SLIDING_TRAIN_YEARS": "nope"},
                         capture_output=True, text=True, cwd=REPO)
    assert bad.returncode != 0, "invalid SLIDING_TRAIN_YEARS did not raise"
    print("[OK] 3. config override: reaches CFG; invalid value raises")

    # 4) launcher fail-fast without --candidate (must exit before creating a run dir)
    runs_before = {p.name for p in (REPO / "runs").iterdir()}
    guard = subprocess.run([PY, "run_baseline.py", "--label", "guard-check"],
                           env=env, capture_output=True, text=True, cwd=REPO)
    runs_after = {p.name for p in (REPO / "runs").iterdir()}
    assert guard.returncode != 0 and "Refusing to launch" in (guard.stderr + guard.stdout), \
        f"guard did not fire: rc={guard.returncode}"
    assert runs_before == runs_after, "guard fired but a run dir was still created"
    print("[OK] 4. fail-fast guard: non-anchor sliding_train_years without --candidate refused")

    print("\nALL A/B #4 PRE-LAUNCH CHECKS PASS")


if __name__ == "__main__":
    main()
