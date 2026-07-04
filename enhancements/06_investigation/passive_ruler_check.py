"""AGENT C — validate the honest walk-forward ruler with rule-based passive baselines.

Read-only: builds the SAME 25 sliding folds and the SAME cost env as the production
harness, then runs rule-based (NOT trained) policies on each fold's TEST window.
No model training, no NN inference. All folds end < 2024-07-01 (lockbox respected).
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))  # …/enhancements/06_investigation -> repo root
OUT = _HERE  # write per-fold CSVs + summary next to this script
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import CFG
import train_ppo
from data_loader import make_sliding_folds
from env_bracket import BracketTradingEnv
import baselines
from evaluate import full_report
from eval_harness import metric_return_over_mtm_dd

# ── load features via the PRODUCTION path (byte-identical bars/features) ──────
m1, feat, cols = train_ppo._load_decision_features()
folds = make_sliding_folds(
    feat, train_years=CFG.sliding_train_years, val_months=CFG.sliding_val_months,
    test_months=CFG.sliding_test_months, step_months=CFG.sliding_step_months,
    embargo_bars=CFG.split_embargo_bars, lockbox_start=CFG.lockbox_start_date)
print(f"[folds] n={len(folds)}  lockbox={CFG.lockbox_start_date}  "
      f"spread_atr_frac={CFG.spread_atr_frac}  ppy={CFG.periods_per_year}", flush=True)

# LOCKBOX ASSERT: no test bar may touch >= 2024-07-01
lock = pd.Timestamp(CFG.lockbox_start_date, tz="UTC")
for i, (tr, va, te) in enumerate(folds, 1):
    assert te.index.max() < lock, f"fold {i} test crosses lockbox!"
print(f"[lockbox] OK: last test bar = {folds[-1][2].index.max()}  < {lock}", flush=True)


def _slice_m1(decision_df):
    # mirror train_ppo._slice_m1_for_decision_window
    start = decision_df.index.min(); end = decision_df.index.max()
    return m1.loc[(m1.index > start) & (m1.index <= end)].copy()


def build_env(decision_df):
    m1_sl = _slice_m1(decision_df)
    return BracketTradingEnv(
        decision_df, m1_sl, cols,
        sl_atr_multipliers=CFG.sl_atr_multipliers,
        tp_r_multipliers=CFG.tp_r_multipliers,
        initial_equity=CFG.initial_equity,
        risk_fraction=CFG.risk_fraction,
        spread_atr_frac=CFG.spread_atr_frac,
        slippage_atr_frac=CFG.slippage_atr_frac,
        commission_per_trade=CFG.commission_per_trade,
        holding_penalty=CFG.holding_penalty,
        reward_mtm_weight=CFG.reward_mtm_weight,
        randomize_start=False, max_episode_steps=None)


def long_only_policy(env):
    # buy-and-hold-ish: always request long with widest SL/TP bucket
    return np.array([1, len(CFG.sl_atr_multipliers) - 1,
                     len(CFG.tp_r_multipliers) - 1], dtype=int)


POLICIES = {
    "passive_ema_atr": baselines.ema_atr_trend_policy,   # the rule-based ruler check
    "long_only":       long_only_policy,
    "random":          baselines.random_policy,
}

# "Strong-bull" folds are defined by MEASURED gold buy-and-hold, not a hardcoded
# list: a fold is bull iff its raw gold B&H >= +8% (matches 06-...md Agent C, which
# yields folds 10,16,17,18,23). The original mandate's guess {9,25} was wrong by
# measurement — fold 9 B&H is -0.66% (chop), fold 25 is +2.84% (mild) — so deriving
# the flag from data keeps this validator self-consistent on any rerun.
BULL_BH_PCT = 8.0

results = {name: {"rows": [], "equities": [], "trades": []} for name in POLICIES}
gold_bh = []

np.random.seed(0)  # only affects random_policy sampling determinism

for k, (tr, va, te) in enumerate(folds, 1):
    win = f"{te.index.min().date()}->{te.index.max().date()}"
    # gold buy-and-hold over the RAW test close (market fact, permitted)
    c = te["Close"].astype(float)
    bh_ret = (c.iloc[-1] / c.iloc[0] - 1.0) * 100.0
    gold_bh.append(bh_ret)

    for name, pol in POLICIES.items():
        env = build_env(te)
        eq, trades = baselines.run_policy(env, pol)
        rep = full_report(eq, trades, initial_equity=CFG.initial_equity,
                          periods_per_year=CFG.periods_per_year)["value"].to_dict()
        fm = metric_return_over_mtm_dd(eq, CFG.initial_equity)
        results[name]["rows"].append({
            "fold": k, "test": win, "bull": bh_ret >= BULL_BH_PCT,
            "gold_bh_pct": round(bh_ret, 3),
            "ret_pct": rep.get("total_return_pct"),
            "pf": rep.get("profit_factor"),
            "sharpe_trade": rep.get("sharpe_trade"),
            "n_trades": rep.get("n_trades"),
            "max_dd_mtm_pct": rep.get("max_drawdown_mtm_pct"),
            "fold_metric": fm,
        })
        results[name]["equities"].append(eq)
        if len(trades):
            results[name]["trades"].append(trades)
    print(f"[fold {k:02d}] {win}  gold_bh={bh_ret:+.1f}%  "
          f"passive_ret={results['passive_ema_atr']['rows'][-1]['ret_pct']:+.2f}%  "
          f"passive_pf={results['passive_ema_atr']['rows'][-1]['pf']}", flush=True)


def stitch(equities):
    running, parts = CFG.initial_equity, []
    for eq in equities:
        if eq is None or eq.empty or "equity" not in eq:
            continue
        factor = running / CFG.initial_equity
        block = pd.DataFrame({"equity": eq["equity"].astype(float) * factor})
        if "equity_mtm" in eq.columns:
            block["equity_mtm"] = eq["equity_mtm"].astype(float) * factor
        parts.append(block)
        running = float(block["equity"].iloc[-1])
    return pd.concat(parts) if parts else pd.DataFrame(columns=["equity"])


out = {"n_folds": len(folds), "policies": {}}
for name in POLICIES:
    df = pd.DataFrame(results[name]["rows"])
    df.to_csv(os.path.join(OUT, f"passive_{name}_perfold.csv"), index=False)
    stitched = stitch(results[name]["equities"])
    all_tr = (pd.concat(results[name]["trades"], ignore_index=True)
              if results[name]["trades"] else pd.DataFrame())
    oos = full_report(stitched, all_tr, initial_equity=CFG.initial_equity,
                      periods_per_year=CFG.periods_per_year)["value"].to_dict()
    stitched_metric = metric_return_over_mtm_dd(stitched, CFG.initial_equity)
    bull_rows = df[df.bull]
    out["policies"][name] = {
        "stitched_metric": round(float(stitched_metric), 4),
        "stitched_return_pct": round(float(oos.get("total_return_pct", float("nan"))), 3),
        "stitched_pf": round(float(oos.get("profit_factor", float("nan"))), 4),
        "stitched_max_dd_mtm_pct": round(float(oos.get("max_drawdown_mtm_pct", float("nan"))), 3),
        "stitched_sharpe_trade": round(float(oos.get("sharpe_trade", float("nan"))), 4),
        "n_trades": int(oos.get("n_trades", 0)),
        "folds_positive": int((df.ret_pct > 0).sum()),
        "folds_total": len(df),
        "median_fold_ret_pct": round(float(df.ret_pct.median()), 3),
        "bull_folds_ret_pct": {int(r.fold): round(float(r.ret_pct), 3) for r in bull_rows.itertuples()},
        "bull_folds_positive": int((bull_rows.ret_pct > 0).sum()),
        "bull_folds_total": len(bull_rows),
    }

# gold buy-and-hold summary + count of bull folds by BH
gdf = pd.DataFrame({"fold": range(1, len(folds)+1), "gold_bh_pct": gold_bh})
gdf["bull"] = gdf.gold_bh_pct >= BULL_BH_PCT
out["gold_bh"] = {
    "n_bh_positive_folds": int((gdf.gold_bh_pct > 0).sum()),
    "n_folds": len(gdf),
    "bull_folds_gold_bh": {int(r.fold): round(float(r.gold_bh_pct), 3) for r in gdf[gdf.bull].itertuples()},
    "median_gold_bh_pct": round(float(gdf.gold_bh_pct.median()), 3),
}
out["rl_anchor_stitched_metric_median"] = -0.526

print("\n===== SUMMARY =====")
print(json.dumps(out, indent=2))
with open(os.path.join(OUT, "passive_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
