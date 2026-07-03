"""Repro/check for the ATR-relative execution cost model (honest-anchor fix,
2026-07-02; doc 02 §3 / doc 05 P2 direction — cost in ATR terms).

Replaces fixed absolute spread/slippage. Both cost legs are priced at the
trade's ENTRY-bar ATR (stored on the Position), so:
  cost-per-R depends only on the SL bucket, never on the era/regime:
  at SL = 1.0xATR, a 1R TP nets exactly 1 - (spread_frac/2 + slip_frac).

Asserts (synthetic bars, exact arithmetic):
  1. Known-ATR charge: entry/exit/R all match hand-computed values.
  2. Regime independence: identical R at ATR=2 and ATR=20.
  3. Pinned calibration (0.0623 / 0.0030): 1R TP nets 0.96585R, SL -1.03415R.
  4. Entry-ATR pinning: cost does NOT re-price if ATR moves during the trade.

Run:  .venv/bin/python checks/check_atr_cost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_bracket import BracketTradingEnv  # noqa: E402


def _env(atr_by_bar, m1_rows, spread=0.10, slip=0.01):
    closes = [100.0] * max(4, len(atr_by_bar) + 2)
    atrs = list(atr_by_bar) + [atr_by_bar[-1]] * (len(closes) - len(atr_by_bar))
    didx = pd.date_range("2024-01-02 00:00", periods=len(closes), freq="1h")
    ddf = pd.DataFrame({"Open": closes, "High": [c + .1 for c in closes],
                        "Low": [c - .1 for c in closes], "Close": closes,
                        "atr": atrs, "f1": 0.0}, index=didx)
    m1i = pd.DatetimeIndex([t for t, *_ in m1_rows])
    m1 = pd.DataFrame({"Open": [r[1] for r in m1_rows], "High": [r[2] for r in m1_rows],
                       "Low": [r[3] for r in m1_rows], "Close": [r[3] for r in m1_rows]},
                      index=m1i)
    return BracketTradingEnv(
        ddf, m1, ["f1"], sl_atr_multipliers=(1.0,), tp_r_multipliers=(1.0,),
        spread_atr_frac=spread, slippage_atr_frac=slip,
        commission_per_trade=0.0, holding_penalty=0.0, reward_mtm_weight=0.0)


def _tp_trade(atr, spread=0.10, slip=0.01):
    """Enter long at bar0 (close=100, given ATR); TP fills in the interval."""
    t = pd.Timestamp("2024-01-02 00:30")
    # TP = 100 + half + sl_dist; make the M1 bar clear it without touching SL.
    half = (spread / 2 + slip) * atr
    tp = 100 + half + atr
    env = _env([atr], [(t, tp - 0.5 * atr, tp + 0.1, tp - 0.6 * atr)],
               spread=spread, slip=slip)
    env.reset()
    env.step(np.array([1, 0, 0]))
    tr = env.trade_log().iloc[0]
    assert tr["exit_reason"] == "TP"
    return tr


def known_charge_check() -> None:
    atr, spread, slip = 2.0, 0.10, 0.01
    half = (spread / 2 + slip) * atr                      # 0.12
    tr = _tp_trade(atr, spread, slip)
    assert abs(tr["entry_price"] - (100 + half)) < 1e-9   # 100.12
    assert abs(tr["entry_atr"] - atr) < 1e-12             # logged on the trade
    assert abs(tr["exit_price"] - (tr["tp"] - half)) < 1e-9
    expect_r = 1 - half / atr                             # 1 - 0.06 = 0.94
    assert abs(tr["r_mult"] - expect_r) < 1e-9, (tr["r_mult"], expect_r)
    print(f"  [1] known-ATR charge exact (entry 100.12, 1R TP nets {expect_r:.4f}R): PASS")


def regime_independence_check() -> None:
    r_low = _tp_trade(2.0)["r_mult"]
    r_high = _tp_trade(20.0)["r_mult"]
    assert abs(r_low - r_high) < 1e-10, (r_low, r_high)
    print(f"  [2] regime independence: R identical at ATR=2 and ATR=20 ({r_low:.6f}R): PASS")


def calibration_figure_check() -> None:
    from config import CFG
    tr = _tp_trade(6.586, spread=CFG.spread_atr_frac, slip=CFG.slippage_atr_frac)
    net = 1 - (CFG.spread_atr_frac / 2 + CFG.slippage_atr_frac)
    assert abs(tr["r_mult"] - net) < 1e-9
    # SL side: gap-free SL touch loses 1 + half/sl_dist.
    t = pd.Timestamp("2024-01-02 00:30")
    half = (CFG.spread_atr_frac / 2 + CFG.slippage_atr_frac) * 6.586
    sl = 100 + half - 6.586
    env = _env([6.586], [(t, sl + 0.4, sl + 0.5, sl - 0.05)],
               spread=CFG.spread_atr_frac, slip=CFG.slippage_atr_frac)
    env.reset()
    env.step(np.array([1, 0, 0]))
    tr_sl = env.trade_log().iloc[0]
    assert tr_sl["exit_reason"] == "SL"
    assert abs(tr_sl["r_mult"] - (-(1 + half / 6.586))) < 1e-9
    print(f"  [3] pinned calibration: 1R TP nets {net:.5f}R, SL loses "
          f"{1 + half / 6.586:.5f}R (constant across eras): PASS")


def entry_atr_pinning_check() -> None:
    # Enter at ATR=2; ATR jumps to 10 by the manual-close bar. Exit cost must
    # still be priced at the ENTRY ATR (2.0), not the close-bar ATR (10.0).
    t = pd.Timestamp("2024-01-02 00:30")
    env = _env([2.0, 10.0, 10.0], [(t, 100.0, 100.05, 99.95)])  # no bracket touch
    env.reset()
    env.step(np.array([1, 0, 0]))         # open at bar0 (ATR 2)
    env.step(np.array([0, 0, 0]))         # manual close at bar1 close=100 (ATR 10)
    tr = env.trade_log().iloc[0]
    half_entry_atr = (0.10 / 2 + 0.01) * 2.0
    assert tr["exit_reason"] == "manual_close"
    assert abs(tr["exit_price"] - (100 - half_entry_atr)) < 1e-9, tr["exit_price"]
    print("  [4] entry-ATR pinning: mid-trade ATR spike does not re-price cost: PASS")


if __name__ == "__main__":
    print("ATR-relative cost checks:")
    known_charge_check()
    regime_independence_check()
    calibration_figure_check()
    entry_atr_pinning_check()
    print("ATR-relative cost checks: ALL PASS")
