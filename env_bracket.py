from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("Install gymnasium first: pip install gymnasium") from exc


@dataclass
class Position:
    direction: int = 0  # +1 long, -1 short, 0 flat
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    units: float = 0.0
    risk_cash: float = 0.0
    sl_distance: float = 0.0
    tp_r: float = 0.0          # planned TP R-multiple (bracket choice)
    sl_atr_mult: float = 0.0   # planned SL ATR multiplier (bracket choice)
    entry_atr: float = 0.0     # ATR at entry — prices BOTH cost legs (ATR-relative cost)
    bars_in_trade: int = 0


class BracketTradingEnv(gym.Env):
    """Gymnasium bracket-order trading environment.

    One step = one decision bar (e.g. M5). TP/SL detection is simulated inside
    the next interval using M1 execution bars.

    Action: MultiDiscrete([3, n_sl_buckets, n_tp_buckets])
        direction: 0 flat/close, 1 long, 2 short
        sl bucket: ATR multiplier index
        tp bucket: R-multiple index
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        decision_df: pd.DataFrame,
        m1_df: pd.DataFrame,
        feature_cols: List[str],
        sl_atr_multipliers=(1.0, 1.5, 2.0),
        tp_r_multipliers=(1.0, 1.5, 2.0, 3.0),
        initial_equity: float = 10_000.0,
        risk_fraction: float = 0.005,
        # ATR-relative execution cost (2026-07-02, honest-anchor fix like N1):
        # round-trip cost scales with the trade's ENTRY-bar ATR, keeping cost
        # dimensionless across regimes/instruments — a fixed absolute spread
        # mis-charges across a 20y backbone (see config.py calibration note).
        spread_atr_frac: float = 0.0623,
        slippage_atr_frac: float = 0.0030,
        commission_per_trade: float = 0.0,
        holding_penalty: float = 0.00002,
        reward_mtm_weight: float = 0.01,
        # Turnover penalty (Phase-C A/B #1): R-units subtracted from REWARD ONLY on
        # each NEW position open. 0.0 = off = anchor. Never touches equity/PnL/metric.
        turnover_penalty_r: float = 0.0,
        # Flip-aware turnover (Phase-C A/B #3): fraction of turnover_penalty_r charged
        # to a FRESH entry (flips always pay full). 1.0 = A/B #1 flat; 0.0 = tax flips only.
        turnover_entry_frac: float = 1.0,
        # Cost-domain randomization (Phase-C A/B #2): per-episode TRAIN-time cost
        # multiplier m ~ U[1-cost_rand_frac, 1+cost_rand_frac]. 0.0 = off = anchor
        # (no RNG draw). Eval/test build this env with 0.0 so the metric is pinned.
        cost_rand_frac: float = 0.0,
        max_episode_steps: Optional[int] = None,
        randomize_start: bool = False,
    ):
        super().__init__()
        self.decision_df = decision_df.dropna(subset=feature_cols + ["atr"]).copy()
        self.m1_df = m1_df.copy()
        self.feature_cols = list(feature_cols)
        self.sl_atr_multipliers = tuple(sl_atr_multipliers)
        self.tp_r_multipliers = tuple(tp_r_multipliers)
        self.initial_equity = float(initial_equity)
        self.risk_fraction = float(risk_fraction)
        self.spread_atr_frac = float(spread_atr_frac)
        self.slippage_atr_frac = float(slippage_atr_frac)
        self.commission_per_trade = float(commission_per_trade)
        self.holding_penalty = float(holding_penalty)
        self.reward_mtm_weight = float(reward_mtm_weight)
        # Validate the turnover penalty at construction (it is env-var-driven, so
        # a bad value can slip in at launch). A NEGATIVE value would flip the sign
        # of `reward -= turnover_penalty_r` into a churn REWARD — a silent
        # inversion of the mechanism — and a NaN/inf would poison the reward and
        # the policy gradient. Fail loudly rather than train a corrupted run.
        _tp = float(turnover_penalty_r)
        if not np.isfinite(_tp) or _tp < 0.0:
            raise ValueError(
                f"turnover_penalty_r must be a finite, non-negative float (R-units); "
                f"got {turnover_penalty_r!r}")
        self.turnover_penalty_r = _tp
        # Flip-aware fresh-entry weight (A/B #3): finite, non-negative (typically [0,1]).
        _tef = float(turnover_entry_frac)
        if not np.isfinite(_tef) or _tef < 0.0:
            raise ValueError(
                f"turnover_entry_frac must be a finite, non-negative float; got {turnover_entry_frac!r}")
        self.turnover_entry_frac = _tef
        # Cost-domain randomization (A/B #2): validate in [0,1) — >=1 would allow a
        # zero/negative cost draw. The per-episode multiplier lives in _cost_mult
        # (1.0 until reset() draws one, and only when cost_rand_frac>0).
        _cr = float(cost_rand_frac)
        if not np.isfinite(_cr) or not (0.0 <= _cr < 1.0):
            raise ValueError(
                f"cost_rand_frac must be a finite float in [0.0, 1.0); got {cost_rand_frac!r}")
        self.cost_rand_frac = _cr
        self._cost_mult = 1.0
        self.max_episode_steps = max_episode_steps or (len(self.decision_df) - 2)

        self.randomize_start = randomize_start

        # Pre-extract M1 open/high/low as contiguous numpy arrays and cache the
        # sorted DatetimeIndex for O(log n) searchsorted lookups.
        # This replaces the O(n) boolean-mask scan on every step — with 1.5 M
        # M1 bars the speedup is ~100-200x per step.
        # Open is needed for honest gap-through-SL fills (doc 05 N1): if a bar
        # OPENS beyond the stop, the realistic fill is the open, not the stop.
        self._m1_open  = self.m1_df["Open"].to_numpy(dtype=np.float64)
        self._m1_high  = self.m1_df["High"].to_numpy(dtype=np.float64)
        self._m1_low   = self.m1_df["Low"].to_numpy(dtype=np.float64)
        self._m1_index = self.m1_df.index  # DatetimeIndex (sorted, tz-aware)

        self.action_space = spaces.MultiDiscrete([3, len(self.sl_atr_multipliers), len(self.tp_r_multipliers)])

        # Market features + position state.
        self.n_pos_features = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols) + self.n_pos_features,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.randomize_start:
            # Random start so each episode covers a different segment of history.
            # Reserve enough room for at least max_episode_steps remaining bars.
            headroom = max(self.max_episode_steps + 2, 2)
            max_start = max(len(self.decision_df) - headroom, 1)
            self.i = int(self.np_random.integers(0, max_start))
        else:
            self.i = 0
        # Cost-domain randomization (A/B #2): draw a per-episode cost multiplier
        # ONLY when enabled. At cost_rand_frac == 0 no RNG is consumed here, so
        # training is bit-identical to the anchor. U[1-f, 1+f] has mean 1.0, so the
        # measured cost level is held — only its dispersion is learned against.
        if self.cost_rand_frac > 0.0:
            self._cost_mult = float(self.np_random.uniform(1.0 - self.cost_rand_frac,
                                                           1.0 + self.cost_rand_frac))
        else:
            self._cost_mult = 1.0
        self.steps = 0
        self.equity = self.initial_equity
        self.realized_pnl = 0.0
        self.position = Position()
        self.trades: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        obs = self._observation()
        return obs, {}

    def _current_row(self):
        return self.decision_df.iloc[self.i]

    def _current_time(self):
        return self.decision_df.index[self.i]

    def _next_time(self):
        return self.decision_df.index[min(self.i + 1, len(self.decision_df) - 1)]

    def _position_state_features(self) -> np.ndarray:
        row = self._current_row()
        close = float(row["Close"])
        atr = max(float(row["atr"]), 1e-12)
        p = self.position
        if p.direction == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

        unrealized = (close - p.entry_price) * p.units * p.direction
        unrealized_r = unrealized / max(p.risk_cash, 1e-12)
        dist_tp_atr = ((p.tp - close) * p.direction) / atr
        dist_sl_atr = ((close - p.sl) * p.direction) / atr
        return np.array([
            p.direction,
            unrealized_r,
            min(p.bars_in_trade / 100.0, 10.0),
            dist_tp_atr,
            dist_sl_atr,
            p.tp_r,
        ], dtype=np.float32)

    def _observation(self):
        row = self._current_row()
        market = row[self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        pos = self._position_state_features()
        obs = np.concatenate([market, pos]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _half_cost(self, atr: float) -> float:
        """Per-side execution cost in price units at a given ATR: half-spread +
        slippage, both as fractions of ATR, scaled by the per-episode cost-domain
        multiplier (A/B #2). _cost_mult == 1.0 unless training-time randomization is
        enabled, so eval/test cost is exactly the pinned value (× 1.0 is exact)."""
        return (self.spread_atr_frac / 2.0 + self.slippage_atr_frac) * atr * self._cost_mult

    def _entry_price(self, close: float, direction: int, atr: float) -> float:
        return close + direction * self._half_cost(atr)

    def _exit_price(self, price: float, direction: int, entry_atr: float) -> float:
        # Long exits at bid below mid; short exits at ask above mid.
        # Both legs are priced at the ENTRY-bar ATR (stored on the Position),
        # so the M1 fill loop needs no bar-level ATR plumbing.
        return price - direction * self._half_cost(entry_atr)

    def _open_position(self, direction: int, sl_idx: int, tp_idx: int):
        row = self._current_row()
        close = float(row["Close"])
        atr = max(float(row["atr"]), 1e-12)
        sl_atr_mult = self.sl_atr_multipliers[sl_idx]
        sl_dist = max(sl_atr_mult * atr, 1e-8)
        tp_r = self.tp_r_multipliers[tp_idx]
        entry = self._entry_price(close, direction, atr)
        sl = entry - direction * sl_dist
        tp = entry + direction * tp_r * sl_dist
        risk_cash = max(self.equity * self.risk_fraction, 1e-8)
        units = risk_cash / sl_dist

        self.position = Position(
            direction=direction,
            entry_time=self._current_time(),
            entry_price=entry,
            sl=sl,
            tp=tp,
            units=units,
            risk_cash=risk_cash,
            sl_distance=sl_dist,
            tp_r=tp_r,
            sl_atr_mult=sl_atr_mult,
            entry_atr=atr,
            bars_in_trade=0,
        )

    def _close_position(self, exit_price_raw: float, exit_time: pd.Timestamp, reason: str) -> float:
        p = self.position
        if p.direction == 0:
            return 0.0
        exit_price = self._exit_price(exit_price_raw, p.direction, p.entry_atr)
        pnl = (exit_price - p.entry_price) * p.units * p.direction - self.commission_per_trade
        self.equity += pnl
        self.realized_pnl += pnl
        r_mult = pnl / max(p.risk_cash, 1e-12)
        self.trades.append({
            "entry_time": p.entry_time,
            "exit_time": exit_time,
            "direction": p.direction,
            "entry_price": p.entry_price,
            "exit_price": exit_price,
            "sl": p.sl,
            "tp": p.tp,
            "sl_atr_mult": p.sl_atr_mult,   # bracket choice: SL ATR multiplier
            "tp_r_bracket": p.tp_r,          # bracket choice: planned TP R-multiple
            "entry_atr": p.entry_atr,        # ATR that priced this trade's costs

            "units": p.units,
            "pnl": pnl,
            "r_mult": r_mult,                # realized R-multiple (actual outcome)
            "bars_in_trade": p.bars_in_trade,
            "exit_reason": reason,
        })
        self.position = Position()
        return pnl

    def _simulate_m1_until_next_decision(self) -> float:
        """Advance within the next decision interval and close if TP/SL is touched.

        Uses DatetimeIndex.searchsorted (O log n) + pre-cached numpy arrays to
        avoid a full pandas boolean-mask scan on each step (~100-200x faster on
        a multi-year M1 dataset).
        """
        p = self.position
        if p.direction == 0:
            return 0.0

        start = self._current_time()
        end   = self._next_time()

        # Half-open interval (start, end]: "right" excludes start, includes end.
        lo = int(self._m1_index.searchsorted(start, side="right"))
        hi = int(self._m1_index.searchsorted(end,   side="right"))

        realized = 0.0
        for idx in range(lo, hi):
            open_ = self._m1_open[idx]
            high  = self._m1_high[idx]
            low   = self._m1_low[idx]
            p = self.position
            if p.direction == 0:
                break
            if p.direction == 1:
                sl_hit = low  <= p.sl
                tp_hit = high >= p.tp
                # Gap-through-SL (doc 05 N1): if the bar OPENS beyond the stop
                # (weekend reopen / news gap), the realistic fill is the open —
                # a stop cannot execute at a price the market never traded.
                sl_fill = min(open_, p.sl)
            else:
                sl_hit = high >= p.sl
                tp_hit = low  <= p.tp
                sl_fill = max(open_, p.sl)

            # Pessimistic intrabar rule: if both touched, assume SL first.
            if sl_hit:
                reason = "SL_gap" if sl_fill != p.sl else "SL"
                realized += self._close_position(sl_fill, self._m1_index[idx], reason)
                break
            if tp_hit:
                # TP stays filled AT the TP price even when the bar gapped past
                # it in our favor — keeps the house pessimism (mirrors SL-first).
                realized += self._close_position(p.tp, self._m1_index[idx], "TP")
                break
        return realized

    def step(self, action):
        action = np.asarray(action, dtype=int)
        direction_raw, sl_idx, tp_idx = int(action[0]), int(action[1]), int(action[2])
        desired_direction = {0: 0, 1: 1, 2: -1}[direction_raw]

        prev_equity = self.equity
        reward_risk_unit = max(prev_equity * self.risk_fraction, 1e-12)
        row = self._current_row()
        close = float(row["Close"])

        # Handle explicit close or flip at current decision close.
        # Per-open turnover-penalty weight (A/B #3 flip-aware): 0.0 = no open this step;
        # 1.0 = a FLIP/reversal (low conviction — full penalty); turnover_entry_frac
        # = a FRESH entry (higher conviction — reduced/zero penalty).
        open_pen_weight = 0.0
        if self.position.direction != 0:
            current_dir = self.position.direction
            if desired_direction == 0:
                self._close_position(close, self._current_time(), "manual_close")
            elif desired_direction != current_dir:
                self._close_position(close, self._current_time(), "flip_close")
                self._open_position(desired_direction, sl_idx, tp_idx)
                open_pen_weight = 1.0                        # FLIP — full penalty

        # Fresh entry if flat and action wants exposure.
        if self.position.direction == 0 and desired_direction != 0:
            self._open_position(desired_direction, sl_idx, tp_idx)
            open_pen_weight = self.turnover_entry_frac       # FRESH — reduced (A/B #3)

        # Simulate TP/SL using the M1 candles inside this decision interval.
        self._simulate_m1_until_next_decision()

        # Dense mark-to-market reward after interval.
        # Uses the CURRENT bar's close (known at decision time) to avoid lookahead.
        # The primary reward is realized PnL from TP/SL hits above; this is a small
        # shaping term (weight 0.01) that gives the agent a directional signal while
        # the trade is open, without leaking any future price information.
        if self.position.direction != 0:
            self.position.bars_in_trade += 1
            current_close = float(self.decision_df.iloc[self.i]["Close"])
            unrealized = (current_close - self.position.entry_price) * self.position.units * self.position.direction
        else:
            unrealized = 0.0

        reward = (self.equity - prev_equity) / reward_risk_unit
        if self.position.direction != 0:
            reward += (unrealized / max(self.position.risk_cash, 1e-12)) * self.reward_mtm_weight
            reward -= self.holding_penalty
        # Turnover penalty (Phase-C A/B #1 flat / #3 flip-aware): REWARD-ONLY term,
        # charged once per NEW position. A flip pays turnover_penalty_r; a fresh entry
        # pays turnover_penalty_r * turnover_entry_frac (entry_frac=1.0 -> A/B #1 flat;
        # 0.0 -> tax flips only). Guarded so turnover_penalty_r==0 OR weight==0 is a
        # strict no-op — NEVER touches equity, PnL, fills, the trade log, or the metric.
        if self.turnover_penalty_r and open_pen_weight:
            reward -= self.turnover_penalty_r * open_pen_weight

        self.history.append({
            "time": self._current_time(),
            "equity": self.equity,
            # Mark-to-market equity (doc 03 §3.9a): realized + open-position
            # unrealized PnL marked at this bar's close. Realized-only equity
            # understates intra-trade drawdown; risk metrics should use this.
            "equity_mtm": self.equity + unrealized,
            "realized_pnl": self.realized_pnl,
            "position": self.position.direction,
            "close": close,
            "reward": reward,
        })

        self.i += 1
        self.steps += 1
        terminated = self.i >= len(self.decision_df) - 2
        truncated = self.steps >= self.max_episode_steps
        obs = self._observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"equity": self.equity, "n_trades": len(self.trades)}
        return obs, float(reward), terminated, truncated, info

    def equity_curve(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=["time", "equity", "realized_pnl", "position", "close", "reward"])
        return pd.DataFrame(self.history).set_index("time")

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
