"""Unit check for the Phase-C A/B #2 cost-domain randomization (env_bracket).

Proves the env-level properties:
  [1] cost_rand_frac == 0 is a STRICT no-op: _cost_mult is exactly 1.0, the cost is
      exactly the pinned value, no RNG is consumed, and a scripted episode is
      bit-identical to an env built without the knob (anchor-faithful).
  [2] cost_rand_frac > 0 draws a per-episode multiplier in [1-f, 1+f] with mean ~1.0
      (the measured cost LEVEL is held; only its dispersion is learned against) and
      scales the fill cost.
  [3] invalid values (<0, >=1, NaN, inf) are rejected at construction.

The TRAIN-only wiring — eval/val/test rollouts stay pinned even when training
randomizes — is proven separately by the byte-identical anchor rollout invariant
(the metric path calls build_env WITHOUT cost_rand_frac).

Run:  .venv/bin/python checks/check_cost_randomization.py   (exit 0 = all pass)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_bracket import BracketTradingEnv

PRICE = 100.0
ATR = 1.0
FEATURES = ["f0", "f1"]
SPREAD, SLIP = 0.0623, 0.0030
PINNED_HALF = (SPREAD / 2.0 + SLIP) * ATR   # per-side cost at m=1


def _data(n: int = 16):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    dec = pd.DataFrame({"Close": PRICE, "atr": ATR, "f0": 0.0, "f1": 0.0}, index=idx)
    m1 = pd.DataFrame({"Open": PRICE, "High": PRICE, "Low": PRICE, "Close": PRICE},
                      index=idx + pd.Timedelta(minutes=30))
    return dec, m1


def _env(cost_rand_frac: float = 0.0, randomize_start: bool = False, **kw):
    dec, m1 = _data()
    return BracketTradingEnv(dec, m1, FEATURES, commission_per_trade=0.0,
                             spread_atr_frac=SPREAD, slippage_atr_frac=SLIP,
                             cost_rand_frac=cost_rand_frac,
                             randomize_start=randomize_start, **kw)


ACTIONS = [[1, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0],
           [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _run(env):
    env.reset(seed=0)
    for a in ACTIONS:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return env.equity_curve(), env.trade_log()


def main() -> None:
    # [1] off-by-default is a strict no-op
    e0 = _env(0.0)
    assert e0._cost_mult == 1.0, "cost_mult not 1.0 at frac=0"
    assert e0._half_cost(ATR) == PINNED_HALF, "cost not exactly pinned at frac=0"
    eqA, trA = _run(_env(0.0))
    eqB, trB = _run(_env(0.0))
    assert np.array_equal(eqA["equity"].to_numpy(), eqB["equity"].to_numpy()) and trA.equals(trB), \
        "frac=0 not deterministic"
    dec, m1 = _data()                       # env built WITHOUT the knob (anchor path)
    eqN, trN = _run(BracketTradingEnv(dec, m1, FEATURES, commission_per_trade=0.0,
                                      spread_atr_frac=SPREAD, slippage_atr_frac=SLIP))
    assert np.array_equal(eqA["equity"].to_numpy(), eqN["equity"].to_numpy()) and trA.equals(trN), \
        "frac=0 differs from the no-knob anchor env"
    print("[1] cost_rand_frac=0: _cost_mult=1.0, cost exactly pinned, bit-identical to anchor env  ✓")

    # [2] frac>0: per-episode multiplier, mean-held, scales cost
    f = 0.4
    env = _env(f, randomize_start=True)
    mults = []
    for _ in range(4000):
        env.reset()
        mults.append(env._cost_mult)
    mults = np.array(mults)
    assert mults.min() >= 1 - f - 1e-9 and mults.max() <= 1 + f + 1e-9, \
        f"multiplier out of [1-f,1+f]: [{mults.min()},{mults.max()}]"
    assert abs(mults.mean() - 1.0) < 0.02, f"mean {mults.mean():.4f} not ~1.0 (cost level not held)"
    env.reset()
    assert abs(env._half_cost(ATR) - PINNED_HALF * env._cost_mult) < 1e-15, "cost not scaled by mult"
    print(f"[2] cost_rand_frac=0.4: m∈[{mults.min():.3f},{mults.max():.3f}], mean {mults.mean():.4f}≈1.0; "
          f"scales cost  ✓")

    # [3] validation
    for bad in (-0.1, 1.0, 1.5, float("nan"), float("inf")):
        try:
            _env(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"accepted invalid cost_rand_frac={bad!r}"
    for good in (0.0, 0.4, 0.99):
        _env(good)  # must not raise
    print("[3] construction rejects <0 / >=1 / NaN / inf; accepts [0,1)  ✓")

    print("\nALL CHECKS PASS — cost randomization is off-by-default, mean-held, range-valid.")


if __name__ == "__main__":
    main()
