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

**Phase-A checkpoint:** B1–B3 complete (commits `ba948c9`, `7a4c3c2`, `e980627`).
User signed off; decisions ratified: N1 lands in Phase A; N3 goes
fraction/quantile (params to be proposed); push branch for backup.

---

## Task 3a — N1 fix: honest gap-through-SL fills  ✅ (2026-07-01)

**Change (training-affecting BY DESIGN — the honest anchor):** the M1 fill sim
filled SL at the bracket price even when the bar OPENED beyond it
(`env_bracket.py:252` pre-fix), and pre-extracted only High/Low
(`env_bracket.py:83-85` pre-fix). Now: `self._m1_open` is pre-extracted, each M1
bar checks the open first — long SL fills at `min(open, sl)`, short at
`max(open, sl)` — and the gapped price flows into `_close_position` (spread/
slippage logic unchanged). Gap fills are tagged `exit_reason="SL_gap"`. TP still
fills AT the TP price even on a favorable gap (house pessimism, mirrors
SL-first).

**Verification (`checks/check_n1_gap_fills.py`) — ALL PASS:**
- Synthetic gap bars, zero costs → exact R arithmetic: long gap −3R (was −1R),
  short gap −3R, normal SL −1R unchanged, favorable TP gap +1R (not +2R),
  both-hit gap bar → SL-first preserved at the gapped fill.
- Census replay on the real smoke window (365d, always-long + always-short
  scripted passes, 1×ATR stops): **11 SL_gap fills, every one verified
  fill-at-open net of costs; ≈11.6R total was previously under-booked
  (≈1.05R extra loss per gap event)** — the old model flattered exactly as
  doc 05 N1 predicted.

**Impact:** all future training/backtests price gap risk; numbers are NOT
comparable to pre-N1 runs (intended — that is the point of fixing the ruler
before the Phase-B baseline).

---

## Task 3b — MTM equity + trade-based Sharpe (doc 03 §3.9a/b)  ✅ (2026-07-01)

**Changes:**
- `env_bracket.py` — history now records `equity_mtm` (realized + open-position
  unrealized, marked at the bar close) beside the realized-only `equity`.
- `evaluate.py` — `max_drawdown_mtm_pct` in `summarize_equity`; new
  `trade_based_sharpe()` (mean/std of per-trade R × √(annual trade rate)),
  reported as `sharpe_trade` in `full_report`. Legacy `sharpe_like` retained
  for continuity until the metric pin (task 5) is ratified.
- `train_ppo.py` — **checkpoint selection now uses MTM drawdown**
  (`_run_one_episode` picks `equity_mtm` when present — part of the honest
  ruler: selection must not be flattered by realized-only DD); sliding/block
  summary rows gain `*_sharpe_trade` + `*_max_dd_mtm_pct`; the stitched OOS
  curve now carries a chained `equity_mtm` column and the stitched report
  prints both DDs + trade-Sharpe.

**Verification (`checks/check_mtm_equity.py`) — ALL PASS:**
- Synthetic dip-then-TP trade: realized DD **0.00%** (the old flattering
  number) vs MTM DD **−0.40%** — exact to construction.
- `trade_based_sharpe` arithmetic vs hand-computed value; NaN guards (<2 trades).
- Real measurement (smoke best model on its val split, 33 trades): realized
  maxDD **−1.55%** vs MTM **−1.87%** — a 0.32pp understatement even on this
  tiny fixture; the gap grows with hold time and position count.

**Note:** selection-behavior change (MTM DD in the consistency score) is an
intended ruler change; trained weights are unaffected (eval-side only). H1-close
marks still miss intra-bar extremes — doc 05 records this as a lower bound.

---

## Task 3c — Multi-seed evaluation harness  ✅ (2026-07-01)

`eval_harness.py`: `multi_seed_run(run_fn, seeds)` → per-seed frame +
median/IQR/range; gate failures counted and warned LOUDLY but kept in the
distribution (hiding them would bias comparisons). Provisional metric helper
`metric_return_over_mtm_dd` (marked PROVISIONAL pending the task-5 pin).
Default seeds (42–46); cheap ranking = first 3, finalists = 5.

**Verified (`checks/check_multi_seed_harness.py`) — ALL PASS:** synthetic
distribution arithmetic exact (median/quartiles/warning/KeyError); real
integration — two 3k-step trainings on smoke data → deterministic val rollouts
→ distribution (median +0.872, range [+0.425, +1.320] across just 2 seeds of an
identical config — the seed-variance evidence that motivates the harness).

---

## Task 3d — Lockbox-carve mechanism  ✅ (2026-07-01)

`config.py` gains `lockbox_start_date` (default None — the concrete date gets
pinned when the Dukascopy backbone lands); `make_sliding_folds` gains
`lockbox_start` and truncates the frame BEFORE any fold is cut, so no train/
val/test window can ever touch the reserved tail; `train_sliding_walk_forward`
passes `CFG.lockbox_start_date` and announces an active lockbox loudly.

**Verified (`checks/check_lockbox_carve.py`) — ALL PASS**, naive + tz-aware:
without carve the sweep reaches past the cut; with carve every window ends
strictly before it, folds equal the pre-truncated-frame result (equivalence),
and `None` is a proven no-op.

**Regression sweep after all Phase-A changes:** all six check scripts
(B1/B2/B3/N1/MTM/lockbox) pass together.

---

## Phase-A checkpoint 2 — awaiting reviewer sign-off (2026-07-01)

DONE: N1 honest fills, MTM equity + trade-Sharpe, multi-seed harness, lockbox
carve (4 commits). PROPOSED, NOT IMPLEMENTED (pin-by-reasoning rule):
- **N3 gate re-form params** — proposal in the checkpoint report: breadth
  `min_consistent_fold_frac = 0.70` (reproduces the old 4-of-5 exactly at n=5;
  ≈0.8% luck-pass probability under a no-edge null at n=34) + PF floor at the
  **10th percentile ≥ 0.90** (matches old worst-of-5 semantics at n=5;
  tolerates ~3 bad folds of 34) + mean trade-Sharpe > 0 once the metric pin
  lands. Awaiting sign-off.
- **Primary metric + ship threshold** — proposal: median across 5 seeds of
  stitched-OOS return ÷ |stitched MTM maxDD|; ship on ≥ +10% relative median
  improvement with ≥4/5 seed-consistency and no gate regression. Awaiting
  sign-off.
Phase B (baseline) remains blocked on the 0b Dukascopy backbone + these pins.

---

## Task 4a — N3 gate landed + both pins RATIFIED  ✅ (2026-07-01) — PHASE A CLOSED

Reviewer ratified both proposals (with adjustments: trade-based Sharpe as the
gate's third leg; a path-based DD secondary added to the metric; the +10%
margin marked provisional with exactly one noise-calibrated adjustment).

**Gate (implemented, `config.py` + `train_ppo.py`):** breadth
`min_consistent_fold_frac = 0.70` (return>0 AND PF>1; `ceil(0.70×5)=4`
reproduces the old block-scheme 4-of-5), floor = 10th-percentile fold PF ≥
0.90, third leg = mean **trade-based** Sharpe > 0 (`val/test_sharpe_trade`).
Old absolute knobs (`min_consistent_folds`, `gate_worst_fold_min_pf`) removed;
grep confirms no stale references. Stale D2 comment in the config block fixed
in passing (it claimed `run_info.NO_DEPLOY.json`; reality = `NO_DEPLOY.txt` +
`gate_passed`).
> **Discipline (verbatim):** a sound gate rejecting the baseline is a RESULT,
> not a trigger to loosen; re-pin only for mechanical mis-specification, never
> to make a result pass.

**Metric (recorded; supporting metrics implemented):** primary = median-of-5 of
(stitched-OOS return ÷ |stitched-OOS max MTM DD|); ALSO report a path-based DD
(Ulcer / return-over-avg-DD) as a non-gating secondary — implemented as
`ulcer_index_mtm` in `evaluate.py`; ship rule = ≥+10% relative AND ≥4/5 seeds
beat running-best median AND no gate regression; 3 seeds rank / 5 finalists.
The +10% margin is PROVISIONAL — reserve ONE recalibration against the
baseline's measured seed-IQR, once, before any Phase-C A/B (calibrating to
noise, not outcome).

**Verification (`checks/check_n3_gate_reform.py`) — ALL PASS:**
- n=5: 4-of-5 passes / 3-of-5 fails (old breadth reproduced exactly); one
  catastrophic fold (PF 0.5) fails the floor; negative mean trade-Sharpe fails.
  Documented divergence: worst-fold PF 0.85 now PASSES the floor (q10 blends to
  0.97) where the old worst-fold rule failed — accepted when ratifying the
  quantile form (smoothing at tiny n).
- n=34: breadth needs 24 (was: 4, trivial); floor tolerates 3 catastrophic
  folds and rejects 4 (was: one bad fold vetoed all 34); NaN (no-trade) folds
  skipped, not zero-treated.
- Ulcer arithmetic exact (flat→0; [100,90,100]→√(100/3)%).

**PHASE A COMPLETE.** The ruler: honest gate honored downstream (B1), honest
pairing (B2/B3), honest fills (N1), honest risk (MTM DD + trade-Sharpe +
Ulcer), fold-count-invariant gate (N3), multi-seed harness, lockbox mechanism,
and both pins recorded. Next: 0b Dukascopy backbone (critical path) → Phase B
baseline under this ruler.

