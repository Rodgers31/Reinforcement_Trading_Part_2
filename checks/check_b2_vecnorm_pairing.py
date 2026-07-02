"""Repro/check for the B2 fix (doc 01): block walk-forward must evaluate the
BEST checkpoint with ITS OWN vecnorm — not the final in-memory model with the
best checkpoint's vecnorm (a mismatched pair that fed the summary and the gate).

Verifies, using models/smoke/ as a fold fixture (no training run needed):
  1. _load_fold_model() returns the BEST-checkpoint pair: its weights equal
     best_model.zip's and differ from the final model's, and the vecnorm path
     is the best checkpoint's own snapshot.
  2. Static: train_walk_forward now calls _load_fold_model(fold_dir) and the
     old manual model/vecnorm pairing is gone.

Run:  .venv/bin/python checks/check_b2_vecnorm_pairing.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # run_info stores repo-root-relative paths


def pairing_check() -> None:
    import torch
    from stable_baselines3 import PPO

    from train_ppo import _load_fold_model

    fold_dir = "models/smoke"
    run_info = json.loads((ROOT / fold_dir / "run_info.json").read_text())
    assert "best_model_path" in run_info, "smoke run saved no best checkpoint — re-run smoke_test.py"

    model, vecnorm_path = _load_fold_model(fold_dir)

    # (a) vecnorm is the BEST checkpoint's own snapshot.
    assert vecnorm_path.name == "best_model_vecnorm.pkl", vecnorm_path
    assert vecnorm_path == Path(run_info["best_model_vecnorm_path"]), vecnorm_path
    assert vecnorm_path.exists()

    # (b) returned weights == best checkpoint's, != final model's.
    best = PPO.load(run_info["best_model_path"], device="cpu")
    final = PPO.load(run_info["model_path"], device="cpu")
    got, exp, fin = (m.policy.state_dict() for m in (model, best, final))
    assert got.keys() == exp.keys()
    assert all(torch.equal(got[k], exp[k]) for k in got), "loaded weights != best checkpoint"
    assert any(not torch.equal(got[k], fin[k]) for k in got), (
        "best and final identical — fixture can't distinguish the pairing; retrain smoke")
    print("  [1] _load_fold_model returns the best-checkpoint model+vecnorm pair: PASS")


def static_check() -> None:
    src = (ROOT / "train_ppo.py").read_text()
    wf = src[src.index("def train_walk_forward"):src.index("def _load_fold_model")]
    assert "model, vecnorm_path = _load_fold_model(fold_dir)" in wf, \
        "train_walk_forward no longer uses _load_fold_model"
    assert 'best_model_vecnorm_path" in run_info' not in wf, \
        "old manual vecnorm pairing still present in train_walk_forward"
    assert "model, _ = train(" not in wf, \
        "train_walk_forward still binds the final in-memory model from train()"
    print("  [2] train_walk_forward evaluates via _load_fold_model (mismatch removed): PASS")


if __name__ == "__main__":
    print("B2 vecnorm-pairing checks:")
    pairing_check()
    static_check()
    print("B2 vecnorm-pairing checks: ALL PASS")
