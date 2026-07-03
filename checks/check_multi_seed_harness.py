"""Repro/check for the multi-seed evaluation harness (doc 04 Phase A).

  1. Synthetic run_fn → exact median/IQR/min/max, gate-failure counting, and
     the loud not-shippable warning.
  2. Real integration: two tiny PPO trainings (3k steps each) on the smoke
     data, metric = provisional return/|MTM DD| from a deterministic val
     rollout, gate flag read from run_info (absent → None on single-split).

Run:  .venv/bin/python checks/check_multi_seed_harness.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from eval_harness import describe, metric_return_over_mtm_dd, multi_seed_run  # noqa: E402


def synthetic_check() -> None:
    vals = {7: 1.0, 8: 3.0, 9: 2.0, 10: -1.0}
    gates = {7: True, 8: False, 9: None, 10: True}

    def run_fn(seed: int) -> dict:
        return {"metric": vals[seed], "gate_passed": gates[seed]}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = multi_seed_run(run_fn, seeds=(7, 8, 9, 10))

    assert res["n_seeds"] == 4
    assert abs(res["median"] - 1.5) < 1e-12          # median of [-1, 1, 2, 3]
    assert abs(res["min"] - (-1.0)) < 1e-12 and abs(res["max"] - 3.0) < 1e-12
    assert abs(res["q25"] - 0.5) < 1e-12 and abs(res["q75"] - 2.25) < 1e-12
    assert res["n_gate_failed"] == 1                  # only seed 8 (None ≠ failed)
    assert "not shippable" in buf.getvalue()          # loud warning fired
    assert list(res["per_seed"].index) == [7, 8, 9, 10]
    # missing metric key must raise
    try:
        multi_seed_run(lambda s: {"oops": 1.0}, seeds=(1,))
        raise AssertionError("missing metric key not caught")
    except KeyError:
        pass
    print("  [1] synthetic distribution + gate counting + warning: PASS")


def integration_check() -> None:
    import json

    from config import CFG
    CFG.csv_path = Path("data/XAU_USD_M1.csv")
    CFG.time_col = "DateTime"
    CFG.source_tz = "UTC"
    CFG.max_days_for_demo = 365

    import train_ppo

    m1, feature_cols, _tr, va, _te = train_ppo.load_datasets()

    def run_fn(seed: int) -> dict:
        out_dir = f"models/harness_check/seed_{seed}"
        train_ppo.train(total_timesteps=3000, train_episode_steps=1024,
                        eval_freq=1500, n_envs=1, seed=seed, device="cpu",
                        out_dir=out_dir, reveal_test=False)
        run_info = json.loads((ROOT / out_dir / "run_info.json").read_text())
        model, vecnorm_path = train_ppo._load_fold_model(out_dir)
        eq, _trades, _rep = train_ppo._rollout_on_split(
            model, vecnorm_path, m1, feature_cols, va)
        return {"metric": metric_return_over_mtm_dd(eq, CFG.initial_equity),
                "gate_passed": run_info.get("gate_passed")}

    res = multi_seed_run(run_fn, seeds=(42, 43))
    print(" ", describe(res, label="smoke 2-seed integration"))
    assert res["n_seeds"] == 2
    assert res["n_gate_failed"] == 0                  # single-split → no gate key
    assert np.isfinite(res["median"]), "metric not finite — rollout broke"
    per_seed = res["per_seed"]["metric"]
    assert per_seed.notna().all()
    print("  [2] real 2-seed train->rollout->distribution integration: PASS")


if __name__ == "__main__":
    print("Multi-seed harness checks:")
    synthetic_check()
    integration_check()
    print("Multi-seed harness checks: ALL PASS")
