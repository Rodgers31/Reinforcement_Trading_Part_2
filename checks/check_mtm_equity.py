"""Repro/check for doc 03 §3.9a/b: mark-to-market equity + trade-based Sharpe.

  1. Synthetic: a trade that dips -0.8R unrealized before hitting TP must show
     ZERO realized drawdown (the old, flattering number) but a -0.4% MTM
     drawdown — proving the new curve measures intra-trade pain.
  2. trade_based_sharpe arithmetic against a hand-computed value.
  3. Real measurement: roll the smoke best model over its val split and report
     the MTM-vs-realized max-drawdown gap (expect |MTM| >= |realized|).

Run:  .venv/bin/python checks/check_mtm_equity.py
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
from evaluate import drawdown, full_report, trade_based_sharpe  # noqa: E402


def synthetic_check() -> None:
    # 5 hourly bars; entry at bar0 close=100 (atr=1 → SL=99, TP=101, zero costs),
    # bar1 closes 99.2 (dip -0.8R, no bracket touch), TP hits inside (t2, t3].
    didx = pd.date_range("2024-01-02 00:00", periods=5, freq="1h")
    closes = [100.0, 99.2, 100.0, 100.0, 100.0]
    ddf = pd.DataFrame({
        "Open": closes, "High": [c + 0.3 for c in closes],
        "Low": [c - 0.3 for c in closes], "Close": closes,
        "atr": 1.0, "f1": 0.0,
    }, index=didx)
    m1 = pd.DataFrame(
        {"Open": [99.6, 99.2, 100.5], "High": [99.7, 99.4, 101.2],
         "Low": [99.3, 99.1, 100.2], "Close": [99.5, 99.3, 101.0]},
        index=pd.DatetimeIndex(["2024-01-02 00:30", "2024-01-02 01:30",
                                "2024-01-02 02:30"]))
    env = BracketTradingEnv(
        ddf, m1, ["f1"], sl_atr_multipliers=(1.0,), tp_r_multipliers=(1.0,),
        spread_price=0.0, slippage_price=0.0, commission_per_trade=0.0,
        holding_penalty=0.0, reward_mtm_weight=0.0,
    )
    env.reset()
    for _ in range(3):  # long, hold, hold — TP fills in the 3rd interval
        env.step(np.array([1, 0, 0]))
    eq = env.equity_curve()

    assert "equity_mtm" in eq.columns
    # Realized curve never dips (flat 10k until the +50 TP) → realized DD = 0.
    dd_real = float(drawdown(eq["equity"].astype(float)).min())
    # MTM marks the -0.8R dip at bar1 close: 10_000 - 0.8*50 = 9_960 → -0.40%.
    dd_mtm = float(drawdown(eq["equity_mtm"].astype(float)).min())
    assert abs(dd_real) < 1e-12, f"realized DD expected 0, got {dd_real}"
    assert abs(dd_mtm - (-0.004)) < 1e-9, f"MTM DD expected -0.40%, got {dd_mtm:%}"
    assert abs(float(eq['equity_mtm'].iloc[-1]) - float(eq['equity'].iloc[-1])) < 1e-9  # flat at end

    rep = full_report(eq, env.trade_log(), initial_equity=10_000.0,
                      periods_per_year=6_003)["value"]
    assert abs(rep["max_drawdown_pct"] - 0.0) < 1e-9
    assert abs(rep["max_drawdown_mtm_pct"] - (-0.4)) < 1e-6
    assert "sharpe_trade" in rep.index and np.isnan(rep["sharpe_trade"])  # 1 trade → NaN
    print("  [1] synthetic MTM dip: realized DD 0.00%, MTM DD -0.40%: PASS")


def sharpe_math_check() -> None:
    r = pd.Series([1.0, -1.0, 2.0, 0.5, -0.5])
    trades = pd.DataFrame({"r_mult": r})
    n_bars, ppy = 6_003, 6_003  # exactly one year → sqrt(n_trades) scaling
    expect = r.mean() / r.std(ddof=1) * np.sqrt(len(r) / 1.0)
    got = trade_based_sharpe(trades, n_bars=n_bars, periods_per_year=ppy)
    assert abs(got - expect) < 1e-12, (got, expect)
    assert np.isnan(trade_based_sharpe(trades.iloc[:1], n_bars, ppy))  # <2 trades
    assert np.isnan(trade_based_sharpe(pd.DataFrame(), n_bars, ppy))
    print(f"  [2] trade_based_sharpe arithmetic (expect {expect:.4f}): PASS")


def smoke_gap_measurement() -> None:
    from config import CFG
    CFG.csv_path = Path("data/XAU_USD_M1.csv")
    CFG.time_col = "DateTime"
    CFG.source_tz = "UTC"
    CFG.max_days_for_demo = 365

    import json
    from stable_baselines3 import PPO
    import train_ppo

    run_info = json.loads((ROOT / "models/smoke/run_info.json").read_text())
    model = PPO.load(run_info["best_model_path"], device="cpu")
    m1, feature_cols, _tr, va, _te = train_ppo.load_datasets()
    eq, trades, rep = train_ppo._rollout_on_split(
        model, run_info["best_model_vecnorm_path"], m1, feature_cols, va)

    dd_real = rep["max_drawdown_pct"]
    dd_mtm = rep["max_drawdown_mtm_pct"]
    assert np.isfinite(dd_mtm), "MTM DD missing from full_report"
    assert abs(dd_mtm) >= abs(dd_real) - 1e-9, (dd_mtm, dd_real)
    print(f"  [3] smoke val rollout ({len(trades)} trades): realized maxDD "
          f"{dd_real:+.2f}%  vs  MTM maxDD {dd_mtm:+.2f}%  "
          f"(understatement {abs(dd_mtm) - abs(dd_real):.2f} pp): PASS")


if __name__ == "__main__":
    print("MTM-equity / honest-Sharpe checks:")
    synthetic_check()
    sharpe_math_check()
    smoke_gap_measurement()
    print("MTM-equity / honest-Sharpe checks: ALL PASS")
