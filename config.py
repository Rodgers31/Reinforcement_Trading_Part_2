from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ── Timeframe helpers ────────────────────────────────────────────────────────

# Map every common notation (MT4/MT5, shorthand, pandas) to the canonical
# pandas resample rule.  Add rows here to support new timeframes.
_TF_TO_PANDAS: dict[str, str] = {
    # MT4/MT5     shorthand    pandas rule
    "M1":  "1min",  "1m":  "1min",  "1min":  "1min",
    "M5":  "5min",  "5m":  "5min",  "5min":  "5min",
    "M15": "15min", "15m": "15min", "15min": "15min",
    "M30": "30min", "30m": "30min", "30min": "30min",
    "H1":  "1h",    "1h":  "1h",   "1H":    "1h",
    "H4":  "4h",    "4h":  "4h",   "4H":    "4h",
    "D1":  "1D",    "1d":  "1D",   "1D":    "1D",
}

# Approximate trading bars per year for XAUUSD.
# Basis: 23 trading hours/day × 261 trading days/year
# (XAUUSD trades ~Sun 22:00 UTC to Fri 22:00 UTC, minus ~1h daily maintenance)
_BARS_PER_YEAR: dict[str, int] = {
    "1min":  360_180,   # 23 × 60 × 261
    "5min":   72_036,   # 23 × 12 × 261
    "15min":  24_012,   # 23 × 4  × 261
    "30min":  12_006,   # 23 × 2  × 261
    "1h":      6_003,   # 23 × 1  × 261
    "4h":      1_501,   # 23/4    × 261  (≈ 6 bars/day)
    "1D":        261,   # 1       × 261
}


@dataclass
class ProjectConfig:
    # ── Production dataset (switched 2026-07-02, Phase B) ────────────────────
    # Dukascopy XAUUSD M1 BID backbone, 2003-05-05 → 2026-07-02 (7.85M rows),
    # acquired + validated on the 0b track (see enhancements/EXECUTION_LOG.md):
    # splice vs OANDA within spread (median |mid diff| 0.025 vs 0.410 spread),
    # lag-0 aligned, 25 features 0 NaN/inf, leakage guardrail PASS.
    # LEAN convention: DateTime,O,H,L,C,V; UTC; bar-open stamps.
    csv_path: Path = Path("data/XAUUSD_M1_Bid_Dukascopy_2003.05.05_2026.07.02.csv")
    time_col: str = "DateTime"
    source_tz: str = "UTC"

    # MT4/MT5-style M1 exports usually timestamp each row at candle OPEN.
    # Internally the project indexes bars by CLOSE time so a decision timestamp
    # means "this candle is complete and known".
    timestamp_is_bar_open: bool = True

    # Execution stays at M1 for realistic TP/SL fill simulation.
    # decision_timeframe controls the bar at which the RL agent observes and
    # acts.  Use any key from _TF_TO_PANDAS: "M5", "H1", "H4", "D1", etc.
    execution_timeframe: str = "1min"
    decision_timeframe: str = "H1"

    # DEMO MODE: set to None for a full production run.
    # 90 days ≈ 25 K M5 bars / ~2 K H1 bars. Enough for pipeline smoke-tests
    # but too small for a publishable result.
    max_days_for_demo: Optional[int] = None
    # 0b census: 2003 is sparse (~40% density) and 2004-05 ramp; the backbone
    # is fully dense from 2006 (333-365k bars/yr). Training universe starts
    # at the clean start so fill simulation quality is uniform.
    start_date: Optional[str] = "2006-01-01"
    end_date: Optional[str] = None

    # ── Validation scheme ────────────────────────────────────────────────────
    # Two compatible modes share the SAME sealed final-test segment:
    #
    #   (a) Single chronological split  → data_loader.split_train_val_test()
    #       train = first train_frac, val = next val_frac, test = remaining.
    #       Used by run_pipeline.py / view_results.py / final_holdout_eval.py.
    #
    #   (b) Walk-forward (default for RL) → data_loader.make_walk_forward_folds()
    #       The last test_frac of bars is sealed as the final holdout; the
    #       remaining "development" region is cut into (n_folds + 1) equal blocks
    #       whose validation windows roll forward in time.  See train_ppo.train_walk_forward().
    #
    # test_frac is kept equal to (1 - train_frac - val_frac) so the sealed test
    # is byte-for-byte identical in both modes — whichever one produced the model,
    # final_holdout_eval.py evaluates it on the same untouched holdout.
    #
    # Long Bid dataset (May 2003 – May 2026, ~23 years of H1 ≈ 138 K bars):
    #   • single split → ~16 yr train / ~3.5 yr val / ~3.5 yr test
    #   • walk-forward → 5 folds rolling over the first ~19.5 yr, last ~3.5 yr sealed
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1        # sealed final holdout (== 1 - train_frac - val_frac)

    # Walk-forward validation (block scheme — train_ppo.train_walk_forward).
    n_walk_forward_folds: int = 5       # number of rolling (train → val) folds
    walk_forward_anchored: bool = False  # True = expanding train window; False = rolling fixed-size

    # ── Sliding-window walk-forward (realistic retrain simulation) ───────────
    # train_ppo.train_sliding_walk_forward(): each fold trains on
    # `sliding_train_years`, selects its checkpoint on the next
    # `sliding_val_months`, then is judged TRUE out-of-sample on the following
    # `sliding_test_months`. The whole window then slides forward by
    # `sliding_step_months` and repeats — simulating "retrain every step_months,
    # then trade the next test_months live." Stitching every fold's test window
    # gives one continuous out-of-sample equity track record.
    #
    # Because every train window is the SAME calendar length, equal timesteps per
    # fold already means equal passes over the data (no per-fold scaling needed).
    #
    # Note on cost: with a 6-month step over ~23 years you get ~34 folds, i.e.
    # ~34 model trainings. Raise sliding_step_months (e.g. 12) to halve the count.
    sliding_train_years: float = 5.0
    sliding_val_months: int = 6
    sliding_test_months: int = 6
    sliding_step_months: int = 6

    # ── Final lockbox (Phase E) ──────────────────────────────────────────────
    # Tail period excluded from the ENTIRE sliding walk-forward sweep — no fold
    # (train, val, or test) may touch bars at/after this date, so iterating on
    # the dev surface can never contaminate it (doc 03 §7, doc 04 Phase A/E).
    # The concrete date gets pinned when the deep Dukascopy backbone lands;
    # it is revealed ONCE, for the single final chosen system.
    # Format "YYYY-MM-DD" (naive dates are localized to the data's timezone).
    # PINNED 2026-07-02 (ratified): final 24 months reserved — the sweep ends
    # at 2024-07-01; everything after is the one-time Phase-E reveal.
    lockbox_start_date: Optional[str] = "2024-07-01"

    # ── Walk-forward deployment gate (re-formed 2026-07-01, doc 05 N3) ───────
    # After the folds finish, the final fold is promoted to the production slot
    # (models/) with run_info["gate_passed"] recording the verdict; on failure a
    # NO_DEPLOY.txt marker is written and downstream tools refuse to use the
    # model without an explicit --allow-failed-gate override (doc 01 B1).
    #
    # Thresholds are FRACTION/QUANTILE-based so the gate's meaning is invariant
    # to fold count — the old absolute counts (4 folds; worst-fold PF ≥ 0.9)
    # were written for the 5-fold block scheme and silently changed meaning at
    # the sliding default's ~34 folds (breadth became trivial, floor draconian).
    #
    # RATIFIED 2026-07-01, pinned by reasoning BEFORE any baseline existed:
    #   breadth ≥ 70% of folds with return>0 AND PF>gate_min_profit_factor
    #           (ceil(0.70×5)=4 reproduces the old 4-of-5 at n=5; luck-pass
    #            probability under a no-edge null at n=34 ≈ 0.8%)
    #   floor   10th-percentile fold PF ≥ 0.90 (≈ old worst-of-5 semantics at
    #           n=5; tolerates isolated bad regimes, rejects fat left tails)
    #   third   mean TRADE-BASED Sharpe > 0 (honest per-trade unit, doc 03 §3.9b)
    # DISCIPLINE: a sound gate rejecting the baseline is a RESULT, not a trigger
    # to loosen; re-pin only for mechanical mis-specification, never to make a
    # result pass.
    min_consistent_fold_frac: float = 0.70     # breadth: fraction of folds needing return>0 AND PF>gate_min_profit_factor
    gate_min_profit_factor: float = 1.0        # a fold "passes" when its PF exceeds this
    gate_pf_floor_quantile: float = 0.10       # the PF floor applies at this quantile across folds
    gate_pf_floor_value: float = 0.90          # ... and that quantile must be >= this
    gate_require_mean_sharpe_positive: bool = True  # AND mean trade-Sharpe across folds must be > 0

    # Embargo removes bars on each side of every split boundary.
    # EMA-200 (the longest lookback used) retains ~37% of a bar's weight after
    # 200 steps; 200-bar embargo makes cross-boundary leakage negligible.
    split_embargo_bars: int = 200

    # Indicators/features.
    atr_period: int = 14
    rsi_period: int = 14
    warmup_bars: int = 250

    # Bracket choices for the RL action space.
    sl_atr_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0)
    tp_r_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

    # Backtest/account model.
    initial_equity: float = 10_000.0
    risk_fraction: float = 0.005  # 0.5% equity risked per trade.

    # ── ATR-relative execution cost (2026-07-02, honest-anchor fix) ──────────
    # Round-trip cost scales with the trade's ENTRY-bar ATR instead of a fixed
    # absolute price. Why: a fixed 0.20 charged 7.7% of ATR in 2006 but 0.9%
    # in 2026 (8x regime distortion), and at the REAL measured recent spread
    # (0.41) it undercharged the recent era ~2x. ATR-relative keeps cost
    # dimensionless (matches features/reward/brackets and the doc-02/05
    # multi-instrument direction) and makes cost-per-R depend only on the SL
    # bucket, never the regime: at SL=1.0xATR a 1R TP nets a constant
    # 1 - (spread_frac/2 + slip_frac) = 0.9659R.
    # PINNED BY MEASUREMENT (never tuned to a result):
    #   spread_atr_frac   = 0.41 / 6.586 = 0.0623
    #     0.41  = median OANDA XAUUSD spread over 2023-2026 (0b splice census)
    #     6.586 = median H1 ATR(14) over the SAME 2023-2026 window (backbone)
    #   slippage_atr_frac = 0.02 / 6.586 = 0.0030 (old recent-era equivalent)
    # Deprecates the absolute spread_price=0.20 / slippage_price=0.02 knobs.
    spread_atr_frac: float = 0.0623     # round-trip spread as a fraction of entry ATR
    slippage_atr_frac: float = 0.0030   # per-side slippage as a fraction of entry ATR
    commission_per_trade: float = 0.01

    # Reward shaping.
    holding_penalty: float = 0.00002
    reward_mtm_weight: float = 0.01
    # Turnover penalty (Phase-C A/B #1): R-units subtracted from REWARD ONLY on
    # each NEW position opened (fresh entry or the open leg of a flip). 0.0 = off
    # = the ratified anchor. Reward-only — it shapes the policy gradient + the
    # consistency-callback checkpoint selection; it NEVER touches equity, PnL,
    # fills, cost, the trade log, or the equity-based metric (honest ruler).
    turnover_penalty_r: float = 0.0
    # Cost-domain randomization (Phase-C A/B #2): per-episode, the TRAINING env
    # scales its execution cost by m ~ U[1-cost_rand_frac, 1+cost_rand_frac] (mean
    # held at the measured cost) so the policy learns cost-robustness. 0.0 = off =
    # anchor. TRAIN-ONLY: eval/val/test rollouts and the equity-based metric always
    # use the pinned cost — the honest ruler is NEVER randomized.
    cost_rand_frac: float = 0.0

    # ── PPO regularisation (generalisation-first preset) ─────────────────────
    # The knobs that most directly control overfitting. Tune these between runs.
    # Rationale: the thin XAUUSD signal lets PPO memorise the train period if it
    # over-optimises, so this preset trains "softer" — more exploration, fewer
    # passes per rollout, a KL brake, a decaying LR, and explicit weight decay —
    # and leans on the best-checkpoint selector to stop before the late collapse.
    ppo_learning_rate: float = 6e-5
    ppo_lr_schedule: str = "linear"        # "linear" decays LR → 0 over training; "constant" holds it
    ppo_ent_coef: float = 0.03             # entropy bonus: higher = more exploration, less brittle policy
    ppo_n_epochs: int = 5                  # SGD passes per rollout: fewer = less per-batch memorisation
    ppo_clip_range: float = 0.1            # PPO trust region (kept tight)
    ppo_target_kl: Optional[float] = 0.025  # early-stop the epoch loop if policy moves too far (None disables)
    ppo_weight_decay: float = 1e-5         # L2 on the policy/value net (Adam optimizer_kwargs)
    ppo_net_arch: Tuple[int, ...] = (128, 64)  # smaller net regularises the thin signal

    # Baseline search space. Wider brackets reduce churn on noisy intraday gold.
    baseline_threshold_grid: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    baseline_sl_idx_grid: Tuple[int, ...] = (1, 2)
    baseline_tp_idx_grid: Tuple[int, ...] = (2, 3)

    # Plotting.
    plot_bars: int = 500

    # ── Derived / read-only properties ──────────────────────────────────────

    @property
    def pandas_tf(self) -> str:
        """Canonical pandas resample rule for decision_timeframe.

        Accepts MT4/MT5 notation ("H1", "M5", "D1"), shorthand ("1h", "5m"),
        or a pandas rule directly ("1h", "5min").  Raises ValueError for
        unrecognised strings so misconfiguration fails loudly.
        """
        rule = _TF_TO_PANDAS.get(self.decision_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown decision_timeframe '{self.decision_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def pandas_execution_tf(self) -> str:
        """Canonical pandas rule for execution_timeframe."""
        rule = _TF_TO_PANDAS.get(self.execution_timeframe)
        if rule is None:
            raise ValueError(
                f"Unknown execution_timeframe '{self.execution_timeframe}'. "
                f"Recognised values: {sorted(_TF_TO_PANDAS.keys())}"
            )
        return rule

    @property
    def periods_per_year(self) -> int:
        """Trading bars per year at decision_timeframe — used for annualising
        Sharpe, Sortino, Calmar.  Derived from _BARS_PER_YEAR lookup so it
        automatically adjusts when you change decision_timeframe."""
        return _BARS_PER_YEAR.get(self.pandas_tf, 6_003)


CFG = ProjectConfig()

# ── Phase-C A/B knob override (launch-time, no code edit) ────────────────────
# A CANDIDATE run may set turnover_penalty_r via the TURNOVER_PENALTY_R env var
# WITHOUT changing the committed anchor default (0.0). run_baseline passes the
# parent environment to every job subprocess, so all jobs in a run share one
# value; it is stamped into each run_info.json and the run registry for audit.
# Unset env var → CFG stays at the ratified anchor (0.0) → anchor reproduced.
import os as _os

_tp_override = _os.environ.get("TURNOVER_PENALTY_R")
if _tp_override is not None:
    try:
        CFG.turnover_penalty_r = float(_tp_override)
    except ValueError as exc:
        raise ValueError(
            f"Invalid TURNOVER_PENALTY_R={_tp_override!r}; expected a float in "
            f"R-units (e.g. 0.045)") from exc
    # Range/finiteness is enforced canonically in BracketTradingEnv.__init__.

_cr_override = _os.environ.get("COST_RAND_FRAC")
if _cr_override is not None:
    try:
        CFG.cost_rand_frac = float(_cr_override)
    except ValueError as exc:
        raise ValueError(
            f"Invalid COST_RAND_FRAC={_cr_override!r}; expected a float in [0,1) "
            f"(e.g. 0.4 = U[0.6,1.4])") from exc
    # Range/finiteness is enforced canonically in BracketTradingEnv.__init__.
