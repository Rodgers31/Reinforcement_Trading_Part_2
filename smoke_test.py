"""Phase 0a smoke-test (doc 04 §6, track 0a) — PLUMBING ONLY, not a performance test.

Proves the existing single-split pipeline loads the OANDA/LEAN XAU_USD_M1.csv, builds
the 25-feature frame, trains PPO for a tiny budget, and produces an equity/trade log
without error. Performance is NOT interpreted here.

Runtime CFG overrides only — config.py is left untouched (fully reversible).
Data: data/XAU_USD_M1.csv is a copy of
  ../trading_bot/market_mechanics_bot/lean_migration/data/XAU_USD_M1.csv
  (OANDA/LEAN export, UTC, mid; data/ is gitignored — re-copy if missing).
Run:  .venv/bin/python smoke_test.py
"""
from pathlib import Path
import numpy as np
from config import CFG

# ── Smoke-test settings (runtime overrides; config.py unchanged) ───────────────
CFG.csv_path = Path("data/XAU_USD_M1.csv")   # LEAN: DateTime,O,H,L,C,V ; UTC ; mid
CFG.time_col = "DateTime"
CFG.source_tz = "UTC"
CFG.timestamp_is_bar_open = True             # default; LEAN/OANDA stamp bar-open
CFG.max_days_for_demo = 365                  # last ~1y → fast, non-empty single split
# execution_timeframe="1min", decision_timeframe="H1" left at their defaults.

import train_ppo

print("=" * 70)
print("PHASE 0a SMOKE TEST — data diagnostics (before training)")
print("=" * 70)
m1, feat, feature_cols = train_ppo._load_decision_features()
n_nan = int(feat[feature_cols].isna().sum().sum())
n_inf = int(np.isinf(feat[feature_cols].to_numpy(dtype=float)).sum())
print(f"M1 rows           : {len(m1):,}  [{m1.index.min()} -> {m1.index.max()}]  tz={m1.index.tz}")
print(f"H1 feature rows   : {len(feat):,}  [{feat.index.min()} -> {feat.index.max()}]")
print(f"feature count     : {len(feature_cols)}   (expect 25)")
print(f"NaN / inf in feats: {n_nan} / {n_inf}   (expect 0 / 0)")
print(f"features          : {feature_cols}")

print("\n" + "=" * 70)
print("PHASE 0a SMOKE TEST — tiny PPO train->eval (5000 steps, single split)")
print("=" * 70)
model, _ = train_ppo.train(
    total_timesteps=5000,
    train_episode_steps=1024,
    eval_freq=2000,
    n_envs=1,
    seed=42,
    device="cpu",
    out_dir="models/smoke",
    reveal_test=False,   # keep the test split sealed
)
print("\nSMOKE TEST COMPLETE — pipeline ran end-to-end with no error.")
