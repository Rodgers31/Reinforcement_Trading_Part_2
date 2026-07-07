"""Pre-launch checks for A/B #5 (Task 25): hold-horizon commitment head.

Synthetic-data unit tests (no CSV load) + subprocess guard checks:
  1. anchor parity: menu () -> action space (3,3,4), obs 25+6, hold branch dead
  2. enabled: action space (3,3,4,4), obs 25+7
  3. horizon fires at exactly k bars, reason 'horizon_close', manual-close
     pricing, hold_bars logged; same-bar re-entry works
  4. early manual close still allowed (k is a ceiling)
  5. flip restarts the clock with the NEW entry's chosen k
  6. brackets keep intrabar priority (TP touch inside the interval wins)
  7. remaining-hold obs feature counts down 1.0 -> 0.5 (k=2)
  8. config env-var parse (valid + invalid) in fresh interpreters
  9. run_baseline fail-fast: HOLD_HORIZON_BARS without --candidate refused

Run: .venv/bin/python checks/check_ab5_hold_horizon.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env_bracket import BracketTradingEnv  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")
MENU = (2, 4, 8, 24)
N_FEAT = 25


def make_synthetic(n_bars: int = 60, tp_touch_bar: int | None = None):
    """Flat 100.0 market: ATR=1.0, no bracket ever touched — unless
    tp_touch_bar is set, in which case that decision interval's M1 spikes
    High to 101.2 (through a 1R/1xATR long TP at ~101.03)."""
    idx = pd.date_range("2020-01-06 00:00", periods=n_bars, freq="1h", tz="UTC")
    feature_cols = [f"f{i}" for i in range(N_FEAT)]
    dec = pd.DataFrame(0.0, index=idx, columns=feature_cols)
    dec["Open"] = dec["High"] = dec["Low"] = dec["Close"] = 100.0
    dec["atr"] = 1.0
    m1_idx = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=1),
                           freq="1min", tz="UTC")
    m1 = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                       "Close": 100.0, "Volume": 0.0}, index=m1_idx)
    if tp_touch_bar is not None:
        # spike inside interval (idx[tp_touch_bar], idx[tp_touch_bar+1]]
        t = idx[tp_touch_bar] + pd.Timedelta(minutes=30)
        m1.loc[t, "High"] = 101.2
    return dec, m1, feature_cols


def env_with(menu, tp_touch_bar=None):
    dec, m1, cols = make_synthetic(tp_touch_bar=tp_touch_bar)
    return BracketTradingEnv(dec, m1, cols, hold_horizon_bars=menu,
                             commission_per_trade=0.01)


HOLD_L = np.array([1, 0, 0, 0])      # long, SL 1.0xATR, TP 1R, k=MENU[0]=2
FLAT = np.array([0, 0, 0, 0])


def main() -> None:
    # 1) anchor parity
    env0 = env_with(())
    assert list(env0.action_space.nvec) == [3, 3, 4], env0.action_space
    assert env0.observation_space.shape == (N_FEAT + 6,)
    env0.step(np.array([1, 0, 0]))   # 3-component action still works
    assert env0.position.hold_bars == 0
    print("[OK] 1. anchor parity: (3,3,4) action space, 31-dim obs, hold_bars=0")

    # 2) enabled shapes
    env = env_with(MENU)
    assert list(env.action_space.nvec) == [3, 3, 4, 4], env.action_space
    assert env.observation_space.shape == (N_FEAT + 7,)
    print("[OK] 2. enabled: (3,3,4,4) action space, 32-dim obs")

    # 3) horizon fires at exactly k=2 + same-bar re-entry + pricing
    env = env_with(MENU)
    obs, _ = env.reset()
    env.step(HOLD_L)                 # entry at bar close 100.0 -> bars=1
    obs, _, _, _, _ = env.step(HOLD_L)  # hold -> bars=2
    assert env.position.bars_in_trade == 2 and len(env.trades) == 0
    env.step(HOLD_L)                 # top-of-step flatten, then same-bar re-entry
    assert len(env.trades) == 1, env.trades
    t = env.trades[0]
    assert t["exit_reason"] == "horizon_close" and t["bars_in_trade"] == 2 \
        and t["hold_bars"] == 2, t
    half = (env.spread_atr_frac / 2 + env.slippage_atr_frac) * 1.0
    assert abs(t["entry_price"] - (100.0 + half)) < 1e-9
    assert abs(t["exit_price"] - (100.0 - half)) < 1e-9
    exp_pnl = (t["exit_price"] - t["entry_price"]) * t["units"] - 0.01
    assert abs(t["pnl"] - exp_pnl) < 1e-9
    assert env.position.direction == 1 and env.position.bars_in_trade == 1 \
        and env.position.hold_bars == 2      # re-entered, fresh clock
    print("[OK] 3. horizon_close at exactly k=2, manual-close pricing, same-bar re-entry")

    # 4) early manual close allowed
    env = env_with(MENU)
    env.reset()
    env.step(np.array([1, 0, 0, 1]))  # k=4
    env.step(FLAT)
    assert env.trades[-1]["exit_reason"] == "manual_close" \
        and env.trades[-1]["bars_in_trade"] == 1
    print("[OK] 4. early manual close allowed (k is a ceiling)")

    # 5) flip restarts the clock with the new k
    env = env_with(MENU)
    env.reset()
    env.step(HOLD_L)                          # long, k=2
    env.step(np.array([2, 0, 0, 3]))          # flip to short, k=24
    assert env.trades[-1]["exit_reason"] == "flip_close"
    assert env.position.direction == -1 and env.position.hold_bars == 24
    for _ in range(3):                        # a k=2 clock would have fired
        env.step(np.array([2, 0, 0, 3]))
    assert env.position.direction == -1 and len(env.trades) == 1
    print("[OK] 5. flip restarts the clock with the NEW entry's k")

    # 6) brackets keep intrabar priority
    env = env_with(MENU, tp_touch_bar=0)      # TP spike inside first interval
    env.reset()
    env.step(np.array([1, 0, 0, 3]))          # long, TP 1R ~ 101.03, k=24
    assert len(env.trades) == 1 and env.trades[0]["exit_reason"] == "TP", env.trades
    print("[OK] 6. brackets fire first inside the interval (TP beat the horizon)")

    # 7) remaining-hold obs feature
    env = env_with(MENU)
    env.reset()
    obs, *_ = env.step(HOLD_L)                # bars=1, k=2 -> remaining 0.5
    assert abs(float(obs[-1]) - 0.5) < 1e-6, obs[-1]
    print("[OK] 7. remaining-hold obs = 0.5 after 1 of 2 bars")

    # 8) config env-var parse in fresh interpreters
    ok = subprocess.run(
        [PY, "-c", "from config import CFG; print(CFG.hold_horizon_bars)"],
        env={**os.environ, "HOLD_HORIZON_BARS": "2,4,8,24"},
        capture_output=True, text=True, cwd=REPO)
    assert ok.returncode == 0 and ok.stdout.strip() == "(2, 4, 8, 24)", ok.stdout
    bad = subprocess.run([PY, "-c", "import config"],
                         env={**os.environ, "HOLD_HORIZON_BARS": "abc"},
                         capture_output=True, text=True, cwd=REPO)
    assert bad.returncode != 0
    print("[OK] 8. config override parses; invalid value raises")

    # 9) launcher fail-fast
    runs_before = {p.name for p in (REPO / "runs").iterdir()}
    guard = subprocess.run([PY, "run_baseline.py", "--label", "guard-check"],
                           env={**os.environ, "HOLD_HORIZON_BARS": "2,4,8,24"},
                           capture_output=True, text=True, cwd=REPO)
    assert guard.returncode != 0 and "Refusing to launch" in (guard.stderr + guard.stdout)
    assert runs_before == {p.name for p in (REPO / "runs").iterdir()}
    print("[OK] 9. fail-fast guard: HOLD_HORIZON_BARS without --candidate refused")

    print("\nALL A/B #5 PRE-LAUNCH CHECKS PASS")


if __name__ == "__main__":
    main()
