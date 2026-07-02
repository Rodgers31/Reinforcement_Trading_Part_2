# Execution log

Chronological record of what was actually run during execution of the doc-04 plan,
its results, and any deviation. One section per task. Facts are cited to `file:line`
or to the command/output that produced them.

---

## Task 0 — Environment + git snapshot  ✅ (2026-07-01)

**Machine:** darwin (arm64), Python **3.9.6** (Xcode system interpreter).

**RL stack:** was missing at session start (`gymnasium`, `stable_baselines3`,
`torch`, `plotly` all `ModuleNotFoundError`; only `numpy 2.0.2` / `pandas 2.3.3` /
`matplotlib 3.9.4` present) — matches the historical env note.

**Action:** created a project venv and installed the pinned-by-`>=` requirements:
```
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```
**Resolved versions (import-verified in the venv):**
python 3.9.6 · gymnasium **1.1.1** · stable-baselines3 **2.7.1** · torch **2.8.0** ·
numpy 2.0.2 · pandas 2.3.3 · plotly 6.8.0 · scikit-learn 1.6.1.

> ⚠️ **Watch-item for Task 1:** these are *newer* than the versions the repo code was
> written against (requirements use `>=`, so pip took the latest). gymnasium 1.x /
> sb3 2.7 / numpy 2.0 have known API drift vs. earlier releases. The Task-1 smoke-test
> is the first thing that will actually exercise this — treat any failure there as a
> possible version-compat issue, not necessarily a code bug. Consider capturing a
> `pip freeze` lockfile once the smoke-test passes, for reproducibility.

**Git snapshot:** branch `main`, single commit `7615a58 "commit 1"`, working tree
clean except `?? enhancements/` (untracked) — the whole plan corpus (01–04 + README)
is not yet version-controlled. `.venv/` is now also untracked (must be git-ignored).

**Task-1 data confirmed on disk:**
`../trading_bot/market_mechanics_bot/lean_migration/data/XAU_USD_M1.csv` — 69 MB,
**1,216,345** M1 rows (LEAN format `DateTime,Open,High,Low,Close,Volume`, UTC, mid).

**Checkpoint:** awaiting user OK on the branch/commit proposal before any code change.
**Resolved:** user approved; branch `execution/phase-a` created, commit `e6bc8ad`
(plan docs + EXECUTION_LOG + `.gitignore`). `.venv/` confirmed untracked.

---

## Task 1 — Phase 0a smoke-test on real data  ✅ (2026-07-01)

**Goal:** prove the single-split pipeline trains end-to-end on data we already have —
**plumbing only, performance NOT interpreted** (doc 04 §6 track 0a).

**How:** `smoke_test.py` uses runtime `CFG` overrides and leaves `config.py` untouched
— a deliberate, reversible deviation from "edit config.py" so no smoke settings leak
into the committed baseline. Data copied to `data/XAU_USD_M1.csv` (gitignored).
Settings: `time_col="DateTime"`, `source_tz="UTC"`, `timestamp_is_bar_open=True`,
`max_days_for_demo=365`, exec=1min / decision=H1 (defaults). Ran
`train_ppo.train(total_timesteps=5000, train_episode_steps=1024, eval_freq=2000,
n_envs=1, device="cpu", out_dir="models/smoke", reveal_test=False)`.

**Result — PASS (exit 0; no errors, no deprecation/API-drift warnings):**
- Data: M1 **353,495** rows [2025-06-11 → 2026-06-11], UTC → resampled to
  **5,666 H1** feature bars [2025-06-26 → 2026-06-11].
- Features: **25** exactly, **0 NaN / 0 inf** — matches the documented set.
- PPO trained (5,120 steps, ~1,600 fps); VecNormalize + consistency callback ran;
  **best checkpoint saved at 2,000 steps** (eligible: train_r +4.05 / val_r +2.54);
  `run_info.json` correctly pairs the best model ↔ `best_model_vecnorm.pkl`.
- Post-training eval produced train/val trade logs + an equity HTML; test kept sealed.
- Artifacts in `models/smoke/` (gitignored): model `.zip`, vecnorm `.pkl`, equity
  `.html`, `run_info.json`, `eval_logs/consistency_evals.csv`, `best_model/`.

**Performance NOT interpreted** — 5k steps is untrained; the −7.7% train / +0.7% val
returns are noise, recorded only to prove the eval path runs.

**Version watch-item RESOLVED:** repo code runs unmodified on gymnasium 1.1.1 /
sb3 2.7.1 / numpy 2.0.2 / torch 2.8.0 — no API drift.

**Incidental (doc 01, NOT fixed — Task 2 scope):** the duplicate + mojibake post-eval
print (B5) reproduced verbatim. B1–B3 untouched.

**New deliverable (uncommitted, pending OK):** `smoke_test.py`.

---

## Interlude — skeptical re-review + doc 05 addendum  ✅ (2026-07-01, no code changes)

Between Task 1 and Task 2 the user requested an adversarial review of the whole
plan + code. Every doc-01 claim was re-verified against live source (B1/B2/B3
confirmed; B3 found narrower than documented — `final_holdout_eval.py:92-95` pairs
correctly, the mismatch is notebook-only). The installed SB3 2.7.1 was traced for
env-dependent behavior, and a **gap census** ran on the real LEAN CSV (pure pandas):
183 weekend reopens in 3.5y, median gap 0.33×ATR(H1), 90th pct 1.35×ATR, max
4.3×ATR; 44 gaps > 1×ATR total.

**Output:** `05-review-addendum-and-perception-audit.md` — missed items N1–N8
(headline: N1 gap-through-SL fills; N2 zero-obs truncation bootstrap verified in
SB3 source; N3 gate doesn't scale to ~34 folds; N7 portfolio risk overlay) plus a
perception/learning-flow audit yielding feature candidates P1–P9 (range position,
observable cost, M1 microstructure, gap awareness, retrain-cadence sweep) with a
batch discipline. Items merged into doc 04's phases (05 §4), doc 02 §8–§9, README
index/status refreshed (env note was stale — stack now runs).

**Plan impact:** none to ordering — Task 2 (B1–B3) unchanged. Two new open
decisions added to doc 04 §5 (N1 sequencing; P9 calendar source).

---

## Task 2a — B1 fix: the deploy gate now gates  ✅ (2026-07-01)

**Re-verified against live source before editing:** `_finalize_deployment` always
promotes and records `gate_passed` (`train_ppo.py:747,751-755`); grep confirmed
**zero readers** of the flag in `final_holdout_eval.py`, `training_diagnostics.py`,
or the notebooks. `test_analysis.ipynb`'s stored outputs even show a real
`gate_passed: False` run analyzed silently (Test +70% reported for a non-approved
model) — the bug had already bitten in production use.

**Fix (blocked-by-default, explicit override — never silent either way):**
- `model_artifacts.py` — new `check_gate_approval()` shared guard: key absent
  (no gate ran, e.g. single-split) → proceed; `true` → proceed; `false` → loud
  banner + `SystemExit(2)`, unless explicitly overridden → proceed with loud
  warning.
- `final_holdout_eval.py` — guard fires immediately after `load_run_info`,
  BEFORE the sealed holdout is revealed; new `--allow-failed-gate` CLI flag.
- `training_diagnostics.py` — guard at the production-model load in section [3];
  new `allow_failed_gate` param + `--allow-failed-gate` CLI flag (SystemExit
  propagates past the section's `except (FileNotFoundError, KeyError)`).
- `notebooks/test_analysis.ipynb` (cell `853a550e`) — `ALLOW_FAILED_GATE = False`
  constant + `RuntimeError` guard right after `run_info` load.

**Repro (`checks/check_b1_gate_guard.py`) — ALL PASS:** unit semantics (4 cases);
end-to-end: fabricated `gate_passed:false` run_info against `models/smoke/`
fixtures → `final_holdout_eval.py` exits 2 with BLOCKED banner; with
`--allow-failed-gate` it warns loudly and proceeds past the guard (then stops at
the absent default CSV — a different failure, proving the guard released).
Fixture auto-removed; refuses to overwrite a real `models/run_info.json`.

**Measurement-only:** touches only reporting/eval surfaces; no training code
path. On the default sliding path, behavior changes only when a run's gate
FAILED — which is the fix, not a regression. Trained weights unaffected.

---

## Task 2b — B2 fix: block walk-forward scores the deployable pair  ✅ (2026-07-01)

**Re-verified against live source:** `train_walk_forward` bound the FINAL
in-memory model from `train()` (`train_ppo.py:837`) yet loaded the BEST
checkpoint's vecnorm (`train_ppo.py:852-856`), then fed that mismatched pair to
`evaluate_on_split` → the per-fold summary AND the deployment gate
(`train_ppo.py:899`). The sliding path's `_load_fold_model` was already correct.

**Fix:** the fold evaluation now loads the deployable pair via
`_load_fold_model(fold_dir)` — best (eligible) checkpoint + its own
`best_model_vecnorm.pkl`, falling back to the final pair only when no best
exists — mirroring the sliding path. `train()`'s return value is no longer
bound; the manual path reconstruction is gone.

**Repro (`checks/check_b2_vecnorm_pairing.py`) — ALL PASS**, using
`models/smoke/` as a fold fixture (no training run): (1) `_load_fold_model`
returns weights tensor-equal to `best_model.zip` and different from the final
model, paired with the best checkpoint's own vecnorm path; (2) static — the
walk-forward body uses `_load_fold_model` and the old mismatched binding is gone.

**Measurement-only:** the diff is confined to the post-training evaluation block
inside `train_walk_forward` (after `train()` returns). The training loop is
untouched; the default sliding path never executes this code. Per-fold summary
numbers and the gate decision on the BLOCK scheme may legitimately change —
they now measure the model that would actually ship (that is the fix).

---

## Task 2c — B3 fix: notebook pairs best checkpoint with its own vecnorm  ✅ (2026-07-01)

**Re-verified against live source:** `test_analysis.ipynb` cell `b8bbe8e7` ran all
three BEST_VAL evaluations with `VECNORM_PATH` (the FINAL model's normalization);
`test_analysis_folds.ipynb` already used `BEST_VECNORM_PATH` correctly — the fix
was never backported (doc 01 B3). Obs stats drift across training, so the final
stats mis-normalize the earlier best checkpoint — the notebook's stored headline
numbers (best-val Test +70%) were computed under the wrong normalization.

**Fix (mirrors the folds notebook):** cell `b8bbe8e7` now defines
`BEST_VECNORM_PATH = run_info.get('best_model_vecnorm_path', models/best_model/
best_model_vecnorm.pkl)` and the three `bv_*` calls pass it; `FINAL_MODEL` keeps
`VECNORM_PATH` (its own stats). Everything else in the cell is unchanged.

**Repro (`checks/check_b3_notebook_pairing.py`) — ALL PASS:** static JSON parse
asserts best↔best-vecnorm and final↔final-vecnorm in BOTH notebooks (the folds
notebook doubles as the reference invariant, so a future regression in either
gets caught).

**Regression sweep after all three fixes:** B1 checks re-pass (same notebook
edited twice), B2 checks pass, and `smoke_test.py` re-ran end-to-end with
exit 0 — the training path is byte-identical in behavior (B1/B3 never touch it;
B2's diff is post-training only).

**Phase-A checkpoint:** B1–B3 complete (commits `ba948c9`, `7a4c3c2`, + B3).
STOPPED per plan — the rest of Phase A (MTM equity, multi-seed harness, pinned
metric, gate re-form N3, N1 fill-model decision, lockbox carve) awaits sign-off.

