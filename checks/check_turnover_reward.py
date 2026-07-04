"""Unit check for the Phase-C A/B #1 turnover-aware reward (env_bracket).

Proves the turnover penalty is a REWARD-ONLY term, fully isolated from the honest
(equity-based) ruler:

  1. A synthetic flat-price episode with a scripted action sequence that opens
     exactly K = 4 new positions. Two envs are run on the SAME actions: one with
     turnover_penalty_r = 0 (anchor), one with turnover_penalty_r = P.
  2. ASSERT: equity, equity_mtm, the trade log, and the return/|MTM-DD| metric are
     BIT-IDENTICAL between the two envs (the penalty never touches them).
  3. ASSERT: total reward drops by EXACTLY K·P, and the per-step reward differs by
     exactly -P on the 4 open steps and 0 on every other step.

Run:  .venv/bin/python checks/check_turnover_reward.py   (exit 0 = all pass)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_bracket import BracketTradingEnv
from eval_harness import metric_return_over_mtm_dd

PRICE = 100.0
ATR = 1.0
FEATURES = ["f0", "f1"]
P = 0.05  # turnover penalty (R-units)


def _synthetic_data(n_dec: int = 16):
    """Flat prices + ATR=1 so SL/TP never trigger intrabar: positions change ONLY
    on explicit manual_close / flip actions, making the open count deterministic."""
    idx = pd.date_range("2020-01-01", periods=n_dec, freq="1h", tz="UTC")
    dec = pd.DataFrame(
        {"Close": PRICE, "atr": ATR, "f0": 0.0, "f1": 0.0}, index=idx
    )
    # one M1 bar per decision interval (at :30), flat OHLC = PRICE
    m1_idx = idx + pd.Timedelta(minutes=30)
    m1 = pd.DataFrame(
        {"Open": PRICE, "High": PRICE, "Low": PRICE, "Close": PRICE}, index=m1_idx
    )
    return dec, m1


def _make_env(turnover_penalty_r: float) -> BracketTradingEnv:
    dec, m1 = _synthetic_data()
    return BracketTradingEnv(
        dec, m1, FEATURES,
        sl_atr_multipliers=(1.0, 1.5, 2.0),
        tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        commission_per_trade=0.0,
        turnover_penalty_r=turnover_penalty_r,
        randomize_start=False,
    )


# Scripted actions [dir_raw(0 flat,1 long,2 short), sl_idx, tp_idx].
# Opens on steps 0 (long), 3 (short), 4 (flip->long), 7 (long) => K = 4; all closed.
ACTIONS = [
    [1, 0, 0],  # 0 open long        (open #1)
    [1, 0, 0],  # 1 hold
    [0, 0, 0],  # 2 close
    [2, 0, 0],  # 3 open short       (open #2)
    [1, 0, 0],  # 4 flip -> long     (open #3, closes short)
    [1, 0, 0],  # 5 hold
    [0, 0, 0],  # 6 close
    [1, 0, 0],  # 7 open long        (open #4)
    [0, 0, 0],  # 8 close
    [0, 0, 0],  # 9 flat
    [0, 0, 0],  # 10 flat
    [0, 0, 0],  # 11 flat
]
K_EXPECTED = 4


def _run(env):
    rewards = []
    env.reset()
    for a in ACTIONS:
        _, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(rewards), env.equity_curve(), env.trade_log()


def main() -> None:
    r0, eq0, tr0 = _run(_make_env(0.0))
    rP, eqP, trP = _run(_make_env(P))

    # ── the honest ruler is untouched ──
    assert np.array_equal(eq0["equity"].to_numpy(), eqP["equity"].to_numpy()), \
        "equity changed under the turnover penalty"
    assert np.array_equal(eq0["equity_mtm"].to_numpy(), eqP["equity_mtm"].to_numpy()), \
        "equity_mtm changed under the turnover penalty"
    assert tr0.equals(trP), "trade log changed under the turnover penalty"
    m0 = metric_return_over_mtm_dd(eq0, 10_000.0)
    mP = metric_return_over_mtm_dd(eqP, 10_000.0)
    assert (m0 == mP) or (np.isnan(m0) and np.isnan(mP)), \
        f"metric changed under the turnover penalty: {m0} vs {mP}"
    print(f"[1] equity/equity_mtm/trade-log/metric BIT-IDENTICAL "
          f"(metric={m0:.6f}) — honest ruler untouched  ✓")

    # ── reward drops by exactly K·P, only on open steps ──
    n_trades = len(tr0)
    assert n_trades == K_EXPECTED, f"expected {K_EXPECTED} trades, got {n_trades}"
    diff = rP - r0
    total = float(diff.sum())
    assert abs(total - (-K_EXPECTED * P)) < 1e-12, \
        f"total reward delta {total} != {-K_EXPECTED * P}"
    open_steps = int(np.sum(np.isclose(diff, -P)))
    zero_steps = int(np.sum(np.isclose(diff, 0.0)))
    assert open_steps == K_EXPECTED, f"{open_steps} steps penalised, expected {K_EXPECTED}"
    assert open_steps + zero_steps == len(diff), \
        "some step reward changed by an amount other than 0 or -P"
    print(f"[2] reward reduced by exactly K·P = {K_EXPECTED}×{P} = {-total:.4f} R; "
          f"per-step delta is -{P} on {open_steps} open steps, 0 elsewhere  ✓")

    print("\nALL CHECKS PASS — turnover penalty is reward-only and ruler-isolated.")


if __name__ == "__main__":
    main()
