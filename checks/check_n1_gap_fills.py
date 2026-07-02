"""Repro/check for the N1 fix (doc 05): honest gap-through-SL fills.

A stop cannot execute at a price the market never traded: if an M1 bar OPENS
beyond the stop (weekend reopen / news gap), the fill must be the open, not the
stop price. TP keeps filling AT the TP price (house pessimism, mirrors SL-first).

  1. Synthetic gap-bar scenarios (zero costs → exact R arithmetic):
     long/short SL gap, normal SL, favorable TP gap, both-hit gap.
  2. Census replay on real data (smoke window): always-in scripted policies;
     every SL_gap trade's fill must equal the M1 open (net of costs), and the
     honest-anchor delta (extra R lost vs fill-at-SL) is reported.

Run:  .venv/bin/python checks/check_n1_gap_fills.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from env_bracket import BracketTradingEnv  # noqa: E402


# ── 1. Synthetic scenarios ────────────────────────────────────────────────────

def _make_env(m1_rows):
    """4 hourly decision bars (close=100, atr=1) + caller-supplied M1 bars.

    Zero spread/slippage/commission → entry=100, SL=99 (long) / 101 (short),
    TP=101 (long) / 99 (short) with the 1.0×ATR SL and 1R TP grids.
    """
    didx = pd.date_range("2024-01-02 00:00", periods=4, freq="1h")
    ddf = pd.DataFrame({
        "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0,
        "atr": 1.0, "f1": 0.0,
    }, index=didx)
    m1i = pd.DatetimeIndex([t for t, *_ in m1_rows])
    m1 = pd.DataFrame(
        {"Open": [r[1] for r in m1_rows], "High": [r[2] for r in m1_rows],
         "Low": [r[3] for r in m1_rows], "Close": [r[3] for r in m1_rows]},
        index=m1i)
    return BracketTradingEnv(
        ddf, m1, ["f1"], sl_atr_multipliers=(1.0,), tp_r_multipliers=(1.0,),
        spread_price=0.0, slippage_price=0.0, commission_per_trade=0.0,
        holding_penalty=0.0, reward_mtm_weight=0.0,
    )


def _one_trade(env, action):
    env.reset()
    env.step(np.array(action))
    trades = env.trade_log()
    assert len(trades) == 1, f"expected exactly one trade, got {len(trades)}"
    return trades.iloc[0]


def synthetic_checks() -> None:
    t = pd.Timestamp("2024-01-02 00:30")  # inside (00:00, 01:00]

    # A) LONG SL gap: bar opens at 97, through SL=99 → fill 97, r=-3 (was -1).
    tr = _one_trade(_make_env([(t, 97.0, 97.5, 96.5)]), [1, 0, 0])
    assert tr["exit_reason"] == "SL_gap" and abs(tr["exit_price"] - 97.0) < 1e-9
    assert abs(tr["r_mult"] - (-3.0)) < 1e-9, tr["r_mult"]

    # B) LONG SL normal: opens 99.5 (above SL), trades down through 99 → fill 99, r=-1.
    tr = _one_trade(_make_env([(t, 99.5, 99.6, 98.8)]), [1, 0, 0])
    assert tr["exit_reason"] == "SL" and abs(tr["exit_price"] - 99.0) < 1e-9
    assert abs(tr["r_mult"] - (-1.0)) < 1e-9

    # C) SHORT SL gap: bar opens at 103, through SL=101 → fill 103, r=-3.
    tr = _one_trade(_make_env([(t, 103.0, 103.5, 102.5)]), [2, 0, 0])
    assert tr["exit_reason"] == "SL_gap" and abs(tr["exit_price"] - 103.0) < 1e-9
    assert abs(tr["r_mult"] - (-3.0)) < 1e-9

    # D) LONG TP favorable gap: opens 102.5 past TP=101 → fill AT 101 (pessimism), r=+1.
    tr = _one_trade(_make_env([(t, 102.5, 102.6, 102.4)]), [1, 0, 0])
    assert tr["exit_reason"] == "TP" and abs(tr["exit_price"] - 101.0) < 1e-9
    assert abs(tr["r_mult"] - 1.0) < 1e-9

    # E) Both hit on a gap-down bar → SL-first preserved, gap fill: open=98 → r=-2.
    tr = _one_trade(_make_env([(t, 98.0, 101.5, 97.0)]), [1, 0, 0])
    assert tr["exit_reason"] == "SL_gap" and abs(tr["exit_price"] - 98.0) < 1e-9
    assert abs(tr["r_mult"] - (-2.0)) < 1e-9

    print("  [1] synthetic gap scenarios (A-E): PASS")


# ── 2. Census replay on real data ────────────────────────────────────────────

def census_replay() -> None:
    from config import CFG
    CFG.csv_path = Path("data/XAU_USD_M1.csv")
    CFG.time_col = "DateTime"
    CFG.source_tz = "UTC"
    CFG.max_days_for_demo = 365

    import train_ppo
    m1, feat, feature_cols = train_ppo._load_decision_features()
    m1_open = m1["Open"]

    n_gap, extra_r, checked = 0, 0.0, 0
    for direction in (1, 2):  # always-long pass, then always-short pass
        env = BracketTradingEnv(
            feat, m1, feature_cols,
            sl_atr_multipliers=CFG.sl_atr_multipliers,
            tp_r_multipliers=CFG.tp_r_multipliers,
            initial_equity=CFG.initial_equity, risk_fraction=CFG.risk_fraction,
            spread_price=CFG.spread_price, slippage_price=CFG.slippage_price,
            commission_per_trade=CFG.commission_per_trade,
        )
        env.reset()
        action = np.array([direction, 0, 3])  # tightest SL (1.0×ATR), widest TP (3R)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(action)
        trades = env.trade_log()
        gaps = trades[trades["exit_reason"] == "SL_gap"]
        half_cost = CFG.spread_price / 2.0 + CFG.slippage_price
        for _, tr in gaps.iterrows():
            open_raw = float(m1_open.loc[tr["exit_time"]])
            d = tr["direction"]
            raw_fill = min(open_raw, tr["sl"]) if d == 1 else max(open_raw, tr["sl"])
            expect = raw_fill - d * half_cost           # _exit_price re-applies costs
            assert abs(tr["exit_price"] - expect) < 1e-6, (tr["exit_time"], tr["exit_price"], expect)
            sl_dist = abs(tr["entry_price"] - tr["sl"])
            extra_r += abs(raw_fill - tr["sl"]) / sl_dist   # loss beyond fill-at-SL
            checked += 1
        n_gap += len(gaps)
        print(f"  [2] {'long' if direction == 1 else 'short'} replay: "
              f"{len(trades)} trades, {len(gaps)} gap-through-SL fills")

    assert n_gap > 0, ("no gap-through fills in the replay window — widen "
                       "max_days_for_demo (census says they occur ~monthly)")
    assert checked == n_gap
    print(f"  [2] census replay: {n_gap} SL_gap fills, all fill-at-open verified; "
          f"honest-anchor delta ≈ {extra_r:.2f}R lost beyond fill-at-SL "
          f"(≈{extra_r / n_gap:.2f}R per gap event)")


if __name__ == "__main__":
    print("N1 gap-fill checks:")
    synthetic_checks()
    census_replay()
    print("N1 gap-fill checks: ALL PASS")
