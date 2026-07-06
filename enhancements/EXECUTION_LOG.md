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

---

## Task 5 — 0b Dukascopy backbone acquired + validated  ✅ (2026-07-02)

**Deliverable:** `data/XAUUSD_M1_Bid_Dukascopy_2003.05.05_2026.07.02.csv` —
**7,851,188 M1 rows, 23.2 years, 574 MB, BID** (repo's RL-default stream;
gitignored, as are the per-year raw chunks in `data/dukascopy_raw/`).

**How:** `dukascopy-node` (Node 22; validated first on one week vs OANDA before
the full pull), resumable per-year chunks with cache+retries. Two per-year
artifacts came down PARTIAL on the first pass (2014 truncated at Jan-10, 2018
at Jan-19 — non-empty files that would have silently passed a size-only check;
caught by a full per-chunk coverage sweep) and two years initially failed
outright; direct CDN probes proved the server HAS all 2014–2015 days (HTTP 200
each), so the failures were client/CDN throttling — resolved with patient
retries + `--no-fail-after-retries` + smaller batches. Final sweep: all 24 bid
+ 4 ask chunks healthy, ~310 trading days/yr, no non-calendar gaps.

**Conversion:** `convert_dukascopy_to_lean.py` (committed) — epoch-ms UTC →
LEAN convention (`DateTime,Open,High,Low,Close,Volume`, UTC, bar-open), 0
duplicate timestamps. Drops into the loader with `time_col="DateTime"`,
`source_tz="UTC"`, `timestamp_is_bar_open=True`.

**Validation (`checks/check_backbone_data.py`, committed) — ALL HARD GATES PASS:**
1. **Splice vs OANDA** (2023-01-02→2026-06-11, n=1,213,082 common bars):
   Dukascopy-mid vs OANDA-mid median |diff| = **0.025** (median spread 0.410 —
   16× inside), systematic offset **+0.0000**, p99 = 0.257. Like-for-like,
   agree well within spread → stitching approved.
2. **Timezone/alignment:** lag-sweep minimum at lag 0 (0.025 vs ~2.5–3.5 at
   ±60/120min) — no DST/offset bug.
3. **N6 volume comparability: REJECT.** Full-window Spearman = **0.694** <
   0.70 pre-set bar (a one-week sample read 0.865 — flattering; the full
   window is the honest number). **Verdict: volume features are REJECTED —
   nothing downstream may use cross-vendor Volume as a feature (doc 05 N6).**
4. **Loads + leakage:** full backbone → 139,611 H1 bars → **25 features,
   0 NaN/0 inf**; `leakage_checks` guardrail PASS.
5. **Census:** 2003 sparse (197 days, ~40% density), 2004–2005 ramping,
   **2006→2026 fully dense** (333–365k bars/yr, 309–313 days/yr, longest gaps
   = weekend/holiday only). **Recommended clean start: 2006-01-01 → 20.5
   dense years → ~28–30 sliding folds** (2003–2005 kept in the file but
   flagged density-degraded; sparse M1 mainly coarsens fill detection, which
   SL-first renders more pessimistic, not less).

**DoD (doc 04 §1.4): MET** — ≥15y ✓ (20.5y dense / 23.2y total), loads through
`prepare_feature_frame` ✓, passes `leakage_checks.py` ✓, splice within spread ✓,
N6 verdict recorded ✓. Phase B is now unblocked on data; config still points at
the old default CSV — switching `csv_path` (+ pinning the lockbox date) belongs
to the Phase-B runner, not this chunk.

---

## Task 6 — ATR-relative execution cost (honest-anchor fix)  ✅ (2026-07-02)

**Ratified decision:** cost model → ATR-relative, calibrated from the measured
0.41 recent OANDA spread. Training-changing (like N1); landed BEFORE the
baseline so the anchor isn't cost-optimistic.

**Pinned BY MEASUREMENT (never tuned to a result):**
`spread_atr_frac = 0.41 / 6.586 = 0.0623`, `slippage_atr_frac = 0.02 / 6.586 =
0.0030`, where 0.41 = median OANDA spread over 2023–2026 (0b splice census) and
6.586 = median H1 ATR(14) over the SAME window, computed from the backbone.

**Why (measured era table):** the old fixed 0.20 spread charged **7.7% of ATR
in 2006 but 0.9% in 2026** (8× regime distortion) and, at the real 0.41 recent
spread, **undercharged the recent era ~2×**. ATR-relative keeps cost
dimensionless (features/reward/brackets already are; doc-02/05 multi-instrument
direction) and makes cost-per-R depend only on the SL bucket:
**1R TP now nets a constant 0.96585R; SL loses 1.03415R — every era.**

**Changes:** `env_bracket.py` — `spread_atr_frac`/`slippage_atr_frac` params,
`_half_cost(atr)`, both legs priced at the trade's **entry-bar ATR** (stored as
`Position.entry_atr`, also logged per-trade); `config.py` — knobs + calibration
provenance (absolutes deprecated/removed); pass-throughs updated in
`train_ppo.build_env` + `run_info`, `run_pipeline.py`, all 3 notebooks (5 env
cells), and the zero-cost fixtures in checks (N1 census math now prices
expected fills at per-trade `entry_atr`). Parameter RENAME on purpose: stale
call sites fail loudly instead of silently keeping absolute semantics.

**Verified (`checks/check_atr_cost.py`) — ALL PASS:** hand-exact charge at a
known ATR; **regime independence** (identical R at ATR=2 and ATR=20);
calibration figures (0.96585R / −1.03415R); **entry-ATR pinning** (mid-trade
ATR spike does not re-price the exit). Full regression sweep: all 7 prior
checks + end-to-end smoke train green under the new model.

**Impact:** pre/post-ATR-cost numbers are NOT comparable (intended). The
doc-05 P2 companion (making cost OBSERVABLE to the agent) remains a Phase-C
item — this task changes what the sim charges, not what the agent sees.

---

## Task 7 — Phase B setup: registry, config switch, sizing run  ✅ (2026-07-03)

**B.0 run registry (`0d8bfc8`):** `runs/<stamp>_<gitsha7>_<label>/` with
`registry.json` (git SHA incl. `-dirty` flag, data name+SHA256+span, full CFG
snapshot, seeds, results); committed `runs/INDEX.md` = one line per run + the
"running best" pointer that moves ONLY via the ratified ship rule. Stamp is
second-resolution (crash-retries/multi-seed days must never collide — learned
from the first sizing attempt). Self-test green.

**Config switch (`033e779`):** production dataset = the 0b backbone;
`start_date=2006-01-01` (census clean start); **lockbox pinned 2024-07-01**
(ratified: final 24 months). Guard `checks/check_phaseb_config.py`: 7.26M M1
rows load via CFG, **25 folds** (vs 29 unlocked — the earlier ~28-30 was
pre-lockbox arithmetic), zero lockbox contact, first train 2006-01-16. Note:
fold geometry leaves a ~5.5-month untested strip (2024-01→2024-07) before the
lockbox — full-test-window requirement, not a bug.

**Sizing run (registry entry #1, label `sizing-run`, fold 25/25 = train
2018→2023, TEST 2023-07→2024-01, seed 42, 3M steps, n_envs=4):**
- First attempt FAILED: driver lacked the `__main__` guard that
  `train_ppo.py` documents as required for `n_envs>1` (SubprocVecEnv worker
  bootstrap re-executed the script → FileExistsError + BrokenPipeError).
  Fixed; the future baseline launcher inherits the guard.
- **Timing: 14.1 min train (3,545 env-steps/s), eval ~1s, data load ~8s →
  ~15 min per fold-seed.** Full baseline = 125 fold-seeds ≈ **31h
  sequential**; 3-seed provisional ≈ 19h; 16 logical/12 perf cores allow
  ~2 concurrent jobs (≈17h / ≈10h).
- **Metric + gate emission verified end-to-end** on real data: per-fold
  metric +2.71 (test ret +8.57%, PF 1.27, trade-Sharpe +2.01, MTM DD −3.16%,
  Ulcer 1.26, 202 trades); N3 gate detail lines emitted; registry finalized.
- **⚠ HONEST-RULER FINDING (one fold, one seed — a flag, not a conclusion):**
  across all 20 evaluations, **zero checkpoints were eligible** — the val leg
  was negative at every eval (−1.4R to −43R). `best_model/` was therefore
  never saved and the test result came from the FINAL (unselected) checkpoint
  via the documented fallback. Under the honest ruler (ATR costs + gap fills
  + MTM DD), this fold shows no exploitable val-period edge; the +8.57% test
  is consistent with a long-lean into a rising test window. Implications:
  (a) do NOT trim the 3M budget on "plateau" grounds — there was no
  learning-curve plateau to exploit, and no evidence 3M over-trains;
  (b) the baseline distribution may be far weaker than the old flattering
  numbers suggested — which is exactly what the ruler was built to reveal;
  (c) folds with zero eligible checkpoints contribute their final model to
  the stitched OOS (system-as-is behavior; the baseline measures it as such).

**PROPOSED COMPUTE PLAN (awaiting sign-off — no full run launched):**
3-seed provisional anchor first (75 fold-seeds, ~19h sequential / ~10h at
2-way concurrency), evaluate the never-eligible pattern across folds, then
extend to 5 seeds (+50 fold-seeds) for the ratified metric. Keep 3M
steps/fold. Background job-pool launcher with per-job registry logging.

---

## Task 8 — 3-seed provisional baseline: PRE-DECLARATION + LAUNCH (2026-07-03)

**APPROVED** by reviewer with binding additions, recorded BEFORE launch:

1. **Seeds pre-declared:** the ratified 5-seed set is **{42, 43, 44, 45, 46}**;
   this provisional pass runs the first three **{42, 43, 44}**; the extension
   (on sign-off) adds **{45, 46}** to the SAME parent run. The extension can
   never be seed-picked — the set is fixed here, in advance, and recorded in
   the parent registry entry (`ratified_seed_set`).
2. **Anchor discipline (absolute):** no config, hyperparameter, cost,
   eligibility, or gate changes in response to anything seen during or after
   this run. Observations land in the Phase-C candidates list below — nothing
   else. The baseline measures the system as-is.
3. **Registry:** parent entry `baseline-3seed` + one job record per fold-seed
   (`jobs/fKK_sSEED/job.json` + crash-safe DONE marker; resumable pool —
   interpretation note: per-job entries live as structured job records inside
   the parent so INDEX.md keeps one line per run). The completed baseline
   **initializes the running-best pointer by definition**; `gate_passed`
   records deployability separately — a gate-FAIL baseline is still the anchor.
4. **Per-seed outputs:** stitched OOS curve (equity+MTM), N3 gate verdict +
   detail, pinned metric — labeled **median-of-3 PROVISIONAL** (the ratified
   metric requires 5 seeds).
5. **Diagnostics to report:** eligibility fraction + per-fold map; train/val
   sign patterns across all evals; churn (trades/window, exit-reason mix incl.
   SL_gap); per-fold metric distribution + seed-IQR (feeds the reserved
   ONE-TIME +10%-margin recalibration — flag only, never recalibrate
   unilaterally); wall-clock vs the ~10h estimate.

**Launcher:** `run_baseline.py` (committed) — resumable fold×seed subprocess
pool (2 concurrent × n_envs=4, OMP/MKL threads capped at 2/job as compute
plumbing), per-job provenance, aggregation + diagnostics + INDEX/running-best
updates. Mini-tested end-to-end (2 folds × 2 seeds × 6k steps: pool, crash
markers, resume-skip, aggregation, gate emission, report, INDEX/running-best —
then fully cleaned up and INDEX restored; one ordering bug found+fixed by the
mini test, which is why it exists).

### PR #1 review triage (2026-07-03, mid-baseline — verified before believing)

Copilot raised 3 findings; each was re-verified against live code assuming it
was wrong. Governing constraint: newly spawned pool jobs re-execute
`run_baseline.py` FROM DISK, so mid-run edits to it would make later jobs run
code differing from the `b9bc9d6` sha in the run's provenance — that file is
frozen until the pool finishes.

- **F1 (fd "leak" in `manage()`): severity claim REFUTED.** CPython refcounting
  closes each parent-side log handle when `log` rebinds on the next launch
  (Popen dups the fd for the child and retains no reference) — steady-state
  parent handles ≈ 1, not 75; no OS-limit risk. The `with`-block IS better
  style → **deferred** to the post-run launcher touch (never edit the live
  launcher mid-run for cosmetics).
- **F2 (duplicate `import pandas` in `_stitch`): VERIFIED, cosmetic.** Same
  file, same freeze → **deferred**, batched with F1.
- **F3 (`smoke_test.py` executes at import): VERIFIED, REAL, FIXED NOW.**
  Module-level CFG mutation + training, and the filename matches pytest's
  `*_test.py` collection pattern — future test collection would silently start
  a training run. Everything moved into `main()` behind the `__main__` guard
  (same discipline the sizing failure taught). Verified: import is 1ms with
  zero side effects, CFG untouched. The file is uninvolved in the pool's job
  path — **zero impact on the running baseline**. Full smoke re-run deferred
  to the post-baseline sweep (re-running now would churn the models/smoke
  fixtures the B1/B2 checks use, mid-flight, for no information).

### Phase-C candidates list (observations only — nothing changes now)
- (from sizing) Never-eligible folds fall back to the FINAL checkpoint —
  consider whether fold-level "no eligible checkpoint" should be surfaced as
  its own diagnostic/gate leg in Phase C. (doc 03 §3.8 territory.)
- (from sizing) Both-legs-negative eval pattern under the honest ruler —
  candidate revisit of reward/selection interplay (doc 03 §3.4b) AFTER the
  anchor exists.
- (from baseline) **Churn/flip cost:** 35% of all exits are direction flips
  (each paying a full round trip) at ~2.7 trades/day — turnover reduction is
  a first-order Phase-C lever (doc 03 §3.4 reward design territory).
- (from baseline) **Transfer-not-fit signature:** train+val− is 33% of evals
  vs 15% both-negative — the agent fits train eras but doesn't transfer;
  sharpens the case for doc-03 regularization/representation items over
  capacity increases.

---

## Task 9 — 3-SEED PROVISIONAL BASELINE COMPLETE  ✅ (2026-07-03) — THE ANCHOR

Run `20260703-021710_b9bc9d6_baseline-3seed`: **75/75 jobs, zero failures,
zero retries.** Wall-clock ≈ **7.9h** (vs ~10h estimate; 56,483s total
job-time, median 12.6 min/job at 2-way concurrency).

### Headline (PROVISIONAL median-of-3; ratified metric needs 5 seeds)

| seed | metric (ret/\|MTM DD\|) | stitched ret | MTM maxDD | Ulcer | PF | trade-Sharpe | gate |
|---|---|---|---|---|---|---|---|
| 42 | −0.528 | −33.5% | −63.4% | 31.1 | 0.984 | −0.19 | FAIL (all 3 legs) |
| 43 | −0.616 | −38.7% | −62.8% | 32.6 | 0.979 | −0.24 | FAIL (all 3 legs) |
| 44 | −0.109 | −4.8% | −43.7% | 26.6 | 1.004 | +0.04 | FAIL (all 3 legs) |

**Median metric = −0.5282, seed-IQR = 0.2533. Gates 0/3.** Breadth 8–9/25
folds positive (need 18); q10 fold-PF 0.81–0.85 (need 0.90); mean trade-Sharpe
≤ 0. **The as-is system has no deployable OOS edge under the honest ruler**
across 2012→2024. This is the anchor working as designed — the old stored
+70% test figure was mis-normalized (B3), cost-flattered (~2× underpriced
recent spread), and gap-flattered (N1); the flattery is now gone.
**Running-best pointer initialized to this baseline BY DEFINITION** (gate
verdict recorded separately, per the ratified rule).

### Diagnostics (the binding list)

- **Eligibility:** 64/75 fold-seeds (85%) produced an eligible checkpoint;
  38.8% of all 1,500 evals were eligible. The sizing scare was NOT systemic —
  and fold-25/seed-42 (the exact sizing configuration) reproduced its
  zero-eligible result. The 11 no-eligible fold-seeds scatter across eras
  (2014, 2017, 2018, 2020, 2022, 2023) and seeds — no single-regime cluster.
- **Train/val sign patterns (1,500 evals):** train+val+ 39%, **train+val−
  33%** (the overfit/transfer-failure quadrant), train−val+ 13%, train−val−
  15%. Dominant failure mode = fits-train-doesn't-transfer, not no-fit.
- **Churn:** ~8,300–8,650 trades per seed over the stitched 12.5y OOS
  (≈340 per 6-month window ≈ 2.7/day). Exit mix: SL 37%, **flip_close 35%**,
  TP 17%, manual 10%, **SL_gap 0.42%** (107 events ≈ 2.9/seed-year —
  consistent with the N1 census given position-open frequency).
- **Per-fold metric distribution:** range −0.96 → +5.99. Consistently
  positive eras: folds 1, 8–9, 25 (2012, 2015–16, 2023H2). Broadly negative:
  folds 15–24 (2018→2023H1). Full arrays in `baseline_report.json`.
- **⚑ RECALIBRATION FLAG (flag only — nothing changed):** the pinned +10%
  RELATIVE ship margin is unusable against this anchor: (a) 10% of |−0.528|
  ≈ 0.053, which is ~5× SMALLER than the seed-IQR (0.253) — inside noise;
  (b) relative-% semantics are ill-defined around a negative/near-zero
  median. The reserved ONE-TIME recalibration should convert the margin to
  an absolute delta calibrated to the measured seed-IQR (e.g. ship requires
  median improvement ≥ 1×IQR) — decide at reviewer sign-off, ideally AFTER
  the 5-seed extension measures the final IQR.

### Artifacts
Parent run dir (registry.json with full provenance; per-job records), per-seed
`seed_*_summary.csv` + `seed_*_stitched.csv`, `baseline_report.json`,
INDEX.md line + running-best pointer (committed).

**STOPPED.** Awaiting sign-off: (1) the {45, 46} extension (+50 jobs ≈ +5h
wall) to complete the ratified 5-seed baseline; (2) the one-time ship-margin
recalibration decision. Phase C untouched.

---

## Task 10 — 5-SEED RATIFIED BASELINE (extension {45,46})  ✅ (2026-07-03) — FINAL ANCHOR

Extended the SAME parent run (`--resume --seeds 45,46`): 50 new jobs, 125/125
total, zero failures. Aggregation derived all 5 seeds from disk (launcher fix
`0fde88c` — a pre-flight catch: the naive `--seeds 45,46` would have finalized
a mislabeled median-of-2).

### Ratified anchor (median-of-5)

| seed | metric | stitched ret | MTM maxDD | Ulcer | PF | trade-Sharpe | folds+ | elig | gate |
|---|---|---|---|---|---|---|---|---|---|
| 42 | −0.528 | −33.5% | −63.4% | 31.1 | 0.984 | −0.19 | 8/25 | 20/25 | FAIL |
| 43 | −0.616 | −38.7% | −62.8% | 32.6 | 0.979 | −0.24 | 8/25 | 23/25 | FAIL |
| 44 | −0.109 | −4.8% | −43.7% | 26.6 | 1.004 | +0.04 | 9/25 | 21/25 | FAIL |
| 45 | −0.526 | −33.5% | −63.6% | 38.9 | 0.987 | −0.18 | 9/25 | 20/25 | FAIL |
| 46 | +0.010 | +0.5% | −53.4% | 26.1 | 1.006 | +0.07 | 10/25 | 22/25 | FAIL |

**RATIFIED METRIC = median-of-5 = −0.5260. Seed-IQR = 0.4188. Gates 0/5.**
Sorted seed metrics: [−0.616, −0.528, −0.526, −0.109, +0.010]. The conclusion
is unchanged and now robust: **the as-is system has no deployable OOS edge
under the honest ruler.** Not one seed clears any gate leg; the best seed (46)
merely breaks even (+0.5% over 12.5y at −53% MTM drawdown).

### Diagnostics refresh — 3-seed → 5-seed shift (125 fold-seeds)

| diagnostic | 3-seed | 5-seed | shift |
|---|---|---|---|
| median metric | −0.5282 | −0.5260 | +0.002 (negligible — ROBUST) |
| **seed-IQR** | 0.2533 | **0.4188** | **materially wider** ⚑ |
| eligibility | 85.3% | 84.8% | unchanged |
| train+val− quadrant | 33% | 33% | unchanged (transfer failure persists) |
| flip_close share | 35% | 35% | unchanged |
| SL_gap share | 0.42% | 0.44% | unchanged |

**Only material shift: the seed-IQR nearly doubled**, because seed 46 landed at
break-even (+0.010) while 42/43/45 cluster near −0.55. The median is stable but
seed dispersion is larger than 3 seeds implied — a single initialization can
swing the stitched metric by ~0.6. This directly drives the recalibration below.
Everything else confirms the 3-seed picture verbatim.

### PROPOSAL 1 — one-time ship-margin recalibration (NOT implemented)

The pinned "+10% relative median" margin is unusable against this anchor:
- Arithmetic: 10% × |−0.526| = **0.0526**. The measured seed-IQR is **0.4188**
  → the old margin is **~1/8 of one IQR**, i.e. deep inside seed noise; any
  "win" that small is indistinguishable from a lucky initialization.
- Relative-% is also ill-defined around a negative/near-zero median.

**Proposed replacement (exact wording), calibrated to the final 5-seed IQR:**
> Ship a Phase-C variant iff ALL three hold:
> 1. **MARGIN** — median-of-5(variant) − median-of-5(running-best) ≥ **1 × IQR_baseline**,
>    where IQR_baseline = **0.4188** (this anchor's 5-seed inter-quartile range).
>    Concretely: variant median ≥ −0.526 + 0.419 = **−0.107**.
> 2. **CONSISTENCY** — ≥ 4 of 5 seeds individually beat the running-best median (−0.526). [unchanged]
> 3. **NO GATE REGRESSION** — variant gate-pass-count ≥ running-best's (0/5 now, so non-binding until something passes). [unchanged]

Notes for the reviewer's ONE-TIME decision (this spends the reserved recalibration):
- 1×IQR is deliberately strict — a single quick win is unlikely to clear a 0.42
  jump in one step, so early A/Bs will likely register directional progress
  without shipping. That is the intended behavior (don't ship noise), but if the
  goal is to ratchet smaller wins, **0.5×IQR (0.209 → variant median ≥ −0.317)**
  is the looser alternative.
- Stronger option worth considering: because every A/B reruns the SAME 25 folds ×
  5 seeds, a **paired per-fold test** (Wilcoxon signed-rank on the 125 paired
  metric deltas) has far more power than comparing two 5-number medians, and is
  robust to the coarse 5-point IQR estimate. Could replace or supplement leg 1.

### PROPOSAL 2 — first Phase-C A/B (NOT started)

Two data-driven candidates from the diagnostics: (a) the transfer-failure
quadrant (train+val− 33% — the reason gates fail) and (b) flip-churn (35% of
exits are direction reversals, each a full round-trip under the now-honest cost).

**Recommendation: cost/slippage domain randomization (doc 03 §3.7a) — targeting
transfer-failure.**
- *Mechanism:* per-episode sample the cost multiplier around the pinned fracs
  (e.g. m ~ U[0.6, 1.4] × {spread=0.0623, slip=0.0030}) — **mean held at the
  measured cost** so the A/B isolates robustness, not a cost-level change. Acts
  as a regularizer that penalizes cost-fragile/brittle policies → should raise
  the val-transfer rate (shrink the train+val− quadrant). Secondary: flips are
  cost-sensitive, so churn may fall as a side effect (measurable via flip_close%).
- *Why this one first:* lowest risk of the three quick wins — **no observation-
  shape change** (no new overfitting surface, no collinearity screen), pure
  env-side change → cleanest attribution; it targets the PRIMARY failure
  (generalization); and it de-risks doc-02 multi-instrument (per-instrument cost
  varies) + pairs with doc-05 P2 (observable cost) later.
- *Runner-up:* HTF context features (§3.1a) is the more direct lever for
  flip-churn specifically, but changes obs shape (retrain + collinearity screen)
  — hold as A/B #2 if cost-randomization doesn't move the flip rate.
- *Exact A/B protocol (doc 04 §3):* ONE change only, nothing else touched.
  (i) RANK on 3 seeds {42,43,44}: train variant on all 25 folds × 3 seeds
  (~75 jobs, ~8h), compare its median-of-3 to the running-best's **3-seed**
  median (−0.5282, apples-to-apples). (ii) If it shows real directional gain,
  promote to FINALIST: extend to 5 seeds {42–46} (+50 jobs) and apply the full
  recalibrated ship rule above. (iii) One decision-log row (ship/kill) either
  way. Dev surface only — the 2024-07+ lockbox stays sealed.

**STOPPED after reporting.** Awaiting sign-off on (1) the recalibration wording
+ 1×IQR vs 0.5×IQR vs paired-test, and (2) the first A/B recommendation. Phase C
remains untouched — no variant code written, no config changed.

---

## Task 11 — Phase C BEGINS: trust investigation committed + ship rule RATIFIED  ✅ (2026-07-03)

**Both Task-10 STOP items resolved by the reviewer:**
1. **Ship-margin recalibration → RATIFIED (final; spends the one-time
   recalibration).** Replaces the noise-sized "+10% relative" with the 0.5×IQR
   combination: ship iff (a) Δmedian ≥ **0.21 ABSOLUTE** (0.5 × baseline seed-IQR
   0.4188) AND (b) paired per-(fold,seed) **Wilcoxon signed-rank p<0.01** with
   positive median delta AND (c) **≥4/5 seeds** beat the running-best median AND
   (d) **no gate regression**. Recorded in doc 04 §2 (Phase A). Wired into new
   `ab_report.py`. The +10% relative margin is retired.
2. **First Phase-C A/B = turnover-aware reward** (reviewer's pick over the
   cost-randomization runner-up), targeting investigation **RF-1**: the honest
   baseline has a small GROSS edge that fair cost eats (94–125% of gross) with
   35% of exits being direction-flips (each a full round trip). Goal: cut
   low-conviction turnover so NET moves toward positive.

**Task 0.1 — trust investigation committed (`19b5b6c`).** `06-baseline-trust-
investigation.md` (the −0.526 anchor is TRUSTWORTHY under adversarial attack; the
old +70% = 0.56× bull-beta + cost flattery, no demonstrable net edge) +
`enhancements/06_investigation/` verification scripts (ruler-fairness, 2023
spread, buy-and-hold facts).

**Task 0.2 — `ab_report.py` (A/B comparator + ratified ship-rule evaluator).**
READ-ONLY (does not touch the system under test): reads candidate + anchor
`baseline_report.json` / summaries / per-job trade logs, evaluates the four legs
(3-seed PREVIEW → 5-seed FINALIST), checks gate legs OK→FAIL per seed, and prints
turnover diagnostics — trades/day, flip%, cost-as-%-of-gross, net PnL — that prove
whether the mechanism engaged. Self-test vs the anchor reproduces trades/day
**2.76**, flip **34.9%**, cost **107.8%** of gross (cross-checks the
investigation's independently-derived figures).

**Next:** Task 1 — implement `turnover_penalty_r` (config knob; reward-only entry
penalty in `env_bracket.step`), PROVE penalty=0 reproduces the anchor
bit-identically, unit-test. Task 2 — pin the level (~1× measured real per-trade
cost ≈ 0.045 R) and launch the 3-seed A/B {42,43,44}. Anchor + honest ruler
INVIOLATE; ONE change only; STOP after the 3-seed report.

---

## Task 12 — A/B #1 turnover reward: IMPLEMENTED + PENALTY PINNED + LAUNCHED (2026-07-03)

**Task 1 — implemented (`37a593d`).** `turnover_penalty_r` (config default 0.0 =
anchor; `TURNOVER_PENALTY_R` env override for candidate runs). `env_bracket.step`
subtracts it from `reward` ONCE per NEW position opened (fresh entry OR the open
leg of a flip), guarded so 0.0 is a strict no-op. Threaded through
`train_ppo.build_env` + `run_pipeline`; stamped into `run_info.json`. **Isolation
PROVEN two ways:** (1) unit test `checks/check_turnover_reward.py` — at penalty
0.05 the equity / equity_mtm / trade log / return-over-MTM-DD metric are
BIT-IDENTICAL to penalty 0, while total reward drops by exactly K·P on the K open
steps and 0 elsewhere; (2) invariant rollout — the stored anchor `f01_s42`
checkpoint rolled through the modified env at penalty 0 reproduces the anchor's
`test_trades.csv` + `test_equity.csv` **byte-identically** (md5-equal) and
`fold_metric` to 10 dp (0.6435982085). The honest (equity-based) ruler is
untouched by construction.

**Parity smoke (pre-launch, `--candidate`, penalty 0.045):** end-to-end verified
that the launch env var reaches the job SUBPROCESS — `run_info.turnover_penalty_r
= 0.045`, registry `cfg.turnover_penalty_r = 0.045` — and that `--candidate`
leaves the running-best pointer on the anchor (smoke run + INDEX line reverted).

**PENALTY PINNED — `turnover_penalty_r = 0.045 R` (first shot, pinned BEFORE
launch, no result-tuning).**
- *Anchor to real cost.* The anchor's measured mean real per-trade cost is
  0.0683 / mean(sl_atr_mult 1.632) = **0.0447 R** (median 0.0455), from its actual
  SL-bucket mix (2.0: 46%, 1.5: 35%, 1.0: 19%; 42,234 trades). 0.045 R ≈ **1×**
  that. So the agent internalises the real round-trip spread a SECOND time —
  "felt" cost ≈ 2× real — the textbook filter: only take trades expected to beat
  ~2× cost. Directly targets **RF-1** (cost eats 94–125% of gross via 35%
  flip-churn at ~2.76 trades/day).
- *Scale.* ~0.045 × ~338 trades/episode ≈ 15 R vs a `val_r` signal of ±10–40 R —
  a strong-but-not-idle turnover filter; per-entry it is a gentle 0.045 R nudge
  against ±1–3 R trade outcomes. Expect a substantial (not total) turnover cut.
- *Pre-committed follow-ups (no mid-run tuning):* improves NET & keeps eligibility
  → extend to 5-seed finalist + full ratified ship rule; over-suppresses
  (near-idle, eligibility collapse) → retry at 0.022 (0.5×); NET worse → kill.

**Launch:** `TURNOVER_PENALTY_R=0.045 run_baseline.py --label turnover-p045-3seed
--seeds 42,43,44 --concurrency 2 --candidate` — 25 folds × 3 seeds = 75 jobs, 3M
steps each, ~8h. Same folds/budgets/gate as the anchor; ONLY the reward term
differs. Lockbox (2024-07+) sealed. On completion: `ab_report.py` →
per-seed + median-of-3 vs the −0.526 anchor, paired 75-pair Wilcoxon, turnover
diagnostics (did turnover drop AND net improve?), 4-leg ship-rule PREVIEW.
**STOP after the 3-seed report for sign-off.**

---

## Task 13 — A/B #1 turnover reward: 3-SEED REPORT  ✅ (2026-07-04) — STRONG, but NOT a preview-ship

Run `20260704-034332_53fe6de_turnover-p045-3seed`: **75/75 jobs, 0 failures**,
~5.4h. `turnover_penalty_r=0.045` (parity-verified in `run_info`/registry).
Comparator: `ab_report.py --candidate <run>` → `ab_report.json`.

### Headline — median-of-3 metric **+1.281** vs anchor **−0.526**  (Δ **+1.807**)
Every one of the 3 shared seeds flips from a deep loss to a strong gain:

| seed | metric cand→anchor | RET% cand→anchor | MTM-DD% cand→anchor | PF | trade-Sharpe | nTrades cand→anchor | folds+ |
|---|---|---|---|---|---|---|---|
| 42 | +1.281 ← −0.528 | +30.7 ← −33.5 | −24.0 ← −63.4 | 1.031 ← 0.984 | +0.26 ← −0.19 | 4000 ← 8451 | 15/25 ← 8 |
| 43 | +0.526 ← −0.616 | +18.3 ← −38.7 | −34.9 ← −62.8 | 1.024 ← 0.979 | +0.18 ← −0.24 | 4376 ← 8318 | 15/25 ← 8 |
| 44 | +2.092 ← −0.109 | +69.1 ← −4.8 | −33.1 ← −43.7 | 1.065 ← 1.004 | +0.45 ← +0.04 | 4429 ← 8650 | 11/25 ← 9 |

### Mechanism ENGAGED exactly as hypothesised (RF-1 targeted)
- **Turnover halved:** 1.39 vs 2.77 trades/day (Δ −1.37); trade count ~4,300 vs ~8,500.
- **Flip-churn:** 25.1% vs 35.4% of exits (Δ −10.3 pts).
- **Cost stops eating the edge:** cost = **68%** of gross vs **111%** (Δ −43 pts).
- **Net cash −5,782 → +13,076** (Δ +18,858); PF crosses 1.0 on all seeds; MTM-DD ~halved.
- Not an idleness/ratio artifact: still ~170 trades/fold, DD substantial (−24 to −35%),
  returns genuinely positive. Isolation proof (Task 12) rules out equity leakage — the
  gain is the POLICY dropping low-conviction (net-negative-after-cost) trades.

### Ship-rule PREVIEW — 2/4 legs, **NOT a ship** (as expected at 3 seeds)
- (a) Δmedian ≥ 0.21 : **PASS** (+1.807).
- (b) paired Wilcoxon p<0.01 & median Δ>0 : **FAIL** — p = **0.030** (n=75, median paired
  Δ +0.215): significant at 0.05 but not the ratified 0.01; the per-fold gain is real but
  not yet uniform (several folds still negative).
- (c) ≥4/5 seeds beat running-best : **3/3 beat** −0.526 but the leg needs 4/5 → resolvable
  only at 5 seeds (preview limitation, not a failure of the idea).
- (d) no gate regression : **PASS**.
- **Deployment gate still 0/3.** The **mean-trade-Sharpe>0** leg now PASSES for all 3 seeds
  (was failing); breadth (15/15/11 < 18) and q10 fold-PF (0.74–0.81 < 0.90) still FAIL.
  A large step toward the gate, not through it.

### Decision-log row (doc 04 §3)
| change | seeds | dev-OOS Δmedian | per-fold Wilcoxon | compute | decision | notes |
|---|---|---|---|---|---|---|
| turnover_penalty_r 0→0.045 | 42,43,44 | **+1.807** (cand +1.281) | p=0.030, medianΔ +0.215 (n=75) | 75 jobs ~5.4h | **EXTEND (pending sign-off)** | mechanism confirmed (turnover −50%, cost/gross 111→68%, net +18.9k); ship-rule preview 2/4; gate 0/3 (Sharpe-leg now OK); running-best UNCHANGED |

### Recommendation — **EXTEND to the 5-seed finalist** (awaiting sign-off)
Meets the pre-committed follow-up ("improves NET & keeps eligibility → finalist"): the
mechanism is confirmed, all 3 seeds beat the anchor by a wide margin, eligibility held
(~0.33). The strict legs (Wilcoxon p<0.01, 4/5-seed) can only be settled at 5 seeds.
Protocol: `--resume <run> --seeds 45,46 --candidate` (+50 jobs, ~5h) → re-run `ab_report`
for the full ratified rule. **Do NOT ship on the preview; running-best stays the −0.526
anchor. 5-seed extension launches only on reviewer sign-off.** **STOPPED per instruction.**

---

## Task 14 — A/B #1 turnover reward: 5-SEED FINALIST  ✅ (2026-07-04) — NO SHIP (rule working)

Extended {45,46} into the SAME run (`--resume --candidate`, `TURNOVER_PENALTY_R=0.045`);
125/125 jobs, all `run_info.turnover_penalty_r=0.045` (parity gate verified). Ran on the
committed code that trained all 5 seeds identically (the two review-hardening edits were
applied AFTER the run — see below).

### Finalist — median metric **+1.281** vs anchor **−0.526** (Δ **+1.807**); all 5 seeds beat the anchor
| seed | metric ← anchor | ret% | PF | trade-Sharpe | folds+ | gate |
|---|---|---|---|---|---|---|
| 42 | +1.281 ← −0.528 | +30.7 | 1.031 | +0.264 | 15/25 | FAIL |
| 43 | +0.525 ← −0.616 | +18.3 | 1.024 | +0.183 | 15/25 | FAIL |
| 44 | +2.092 ← −0.109 | +69.1 | 1.065 | +0.447 | 11/25 | FAIL |
| 45 | +0.458 ← −0.526 | +18.2 | 1.024 | +0.179 | 10/25 | FAIL |
| 46 | +2.079 ← +0.010 | +56.5 | 1.048 | +0.386 | 14/25 | FAIL |

### RATIFIED ship rule — 3/4 legs, **SHIP = False**
- (a) Δmedian ≥ 0.21 : **PASS** (+1.807)
- (b) paired Wilcoxon p<0.01 & median Δ>0 : **FAIL — p = 0.0204** (n=125, median Δ +0.209)
- (c) ≥4/5 seeds beat running-best : **PASS (5/5)**
- (d) no gate regression : **PASS**

**Why (b) fails despite the huge aggregate:** paired deltas are **74/125 improved, 51/125
regressed** (mean Δ +0.341 > median +0.209). The lift is carried by large wins on a
majority of folds while ~41% of cells get worse — a favorable *average trade-off*, not a
*broad* improvement. The Wilcoxon leg exists precisely to withhold a ship in this case.
**Deployment gate still 0/5** (breadth 10–15/25 < 18; q10 fold-PF 0.74–0.81 < 0.90; the
mean-trade-Sharpe leg now PASSES all 5 seeds — real progress toward the gate).

### Mechanism (5-seed, vs anchor, all seeds)
trades/day 2.76 → **1.46**; flips 35% → **25.6%**; cost/gross **108% → 69%**; net cash
**−6,902 → +21,679**. RF-1 lever confirmed at scale.

### DECISION — **NO SHIP.** Running-best stays the −0.526 anchor.
The ratified rule requires all four legs; (b) fails at p=0.020. **The pinned p<0.01 is NOT
relaxed** (that would be the forbidden result-driven tuning). This is a *strong-but-not-
broad* outcome — neither the "kill" (net is far better) nor the "over-suppressed→0.022"
(not near-idle; 5/5 beat; eligibility held) pre-committed branch fires, so the next step is
a reviewer decision. Turnover reduction is confirmed as the first-order lever; the flat
0.045 penalty buys aggregate at the cost of per-fold consistency (the one failing dimension).

### PR #2 review (Copilot) — all 5 comments interrogated + addressed
- **env_bracket validate `turnover_penalty_r` finite/non-negative** (VALID, real footgun: a
  negative value inverts the term into a churn REWARD; NaN/inf poisons the gradient) — fixed
  at `__init__`; **neutral at 0.045** (unit test + byte-identical anchor `f01_s42` invariant
  re-verified). Applied AFTER the run so all 125 jobs share identical training code.
- **config: clearer `TURNOVER_PENALTY_R` parse error** (VALID, low-sev) — fixed.
- **passive_ruler_check bull folds** (VALID: stale {9,25} contradicted the measured finding)
  — now derived from measured gold B&H ≥ +8% (self-consistent; no headline changed).
- **06-md / README "nothing touched sealed data"** (VALID precision, NOT a breach) — reworded:
  no *model* train/eval touched the lockbox; raw-price market-fact reads (buy-and-hold) may
  read beyond it. Lockbox intact.

### Decision-log row (doc 04 §3)
| change | seeds | Δmedian | Wilcoxon | gate | decision | notes |
|---|---|---|---|---|---|---|
| turnover_penalty_r 0→0.045 | 42–46 (finalist) | **+1.807** | **p=0.0204 (FAIL <0.01)** | 0/5 | **NO SHIP** | 5/5 seeds beat; mechanism confirmed (turnover −47%, cost/gross 108→69%, net +28.6k); fails per-fold consistency; running-best UNCHANGED |

---

## Task 15 — A/B #1b: turnover level SWEEP 0.022 (0.5×) — PINNED + LAUNCHED (2026-07-04)

**Reviewer sign-off:** after A/B #1 finalist NO-SHIP, sweep the LOWER pre-declared level.
This is a NEW A/B (new run dir, own pin), NOT a re-tune of #1.

**PENALTY PINNED — `turnover_penalty_r = 0.022` (= 0.5× the anchor's measured mean real
per-trade cost 0.0447; the pre-declared alternative level).**
- *Hypothesis (from #1's failure):* A/B #1 (0.045) failed ONLY the paired-Wilcoxon
  consistency leg — 74/125 folds improved but **51/125 regressed** (the flat penalty
  over-suppresses ~41% of folds). Halving the penalty should push fewer folds into the red
  → a MORE UNIFORM improvement that can clear p<0.01, at the cost of a smaller aggregate
  median. The test: does 0.022 raise the improved:regressed ratio vs 0.045's 74:51 while
  still beating the −0.526 anchor?
- *Pre-committed reads (no mid-run tuning):* consistency improves + still beats anchor +
  eligibility held → extend to 5-seed finalist (sign-off); aggregate edge collapses
  (net back toward ≤0) → the flat penalty can't satisfy both aggregate AND consistency →
  stop the level-sweep thread, move to A/B #2 (cost-domain randomization) or a
  conviction-scaled redesign.

**Code parity:** runs on the current review-fixed code (env/config input validation added
after #1). Those edits are proven NEUTRAL at any penalty (byte-identical anchor `f01_s42`
invariant + unit test re-verified), so this is still a clean one-change A/B vs the anchor
(only `turnover_penalty_r`: 0 → 0.022). All 75 jobs share identical committed code.

**Launch:** `TURNOVER_PENALTY_R=0.022 run_baseline.py --label turnover-p022-3seed
--seeds 42,43,44 --concurrency 2 --candidate` — 75 jobs, ~5.4h. On completion: `ab_report`
vs the anchor (ship-rule preview) PLUS a direct 0.022-vs-0.045 consistency comparison.
**STOP after the 3-seed report for sign-off.**

---

## Task 16 — A/B #1b: 0.022 sweep — NO SHIP; flat-penalty level sweep CONCLUDED  ✅ (2026-07-04)

Run `20260704-222152_49a9ef9_turnover-p022-3seed`: 75/75 jobs, all `turnover_penalty_r=0.022`
(parity verified), running-best UNCHANGED. Hypothesis (a lower penalty improves per-fold
consistency) is **REFUTED**.

### Head-to-head vs the anchor (shared 3 seeds {42,43,44}, apples-to-apples)
| level | median-of-3 | improved:regressed | medianΔ | Wilcoxon p | cost/gross | net cash |
|---|---|---|---|---|---|---|
| anchor | −0.5282 | — | — | — | 111% | −5,782 |
| **0.022** | **−0.1687** | **42:33** | +0.080 | **0.4502** | 107% | −2,569 |
| 0.045 | +1.2810 | 44:31 | +0.215 | 0.0304 | 69% | +21,679 |

Per-seed (shared): s42 −0.528→(0.022)−0.030→(0.045)+1.281; s43 −0.616→**−0.750**→+0.525;
s44 −0.109→**−0.169**→+2.092. **At 0.022, seeds 43 & 44 are WORSE than their own anchor.**

### Ship rule (0.022 preview): 2/4 — NO SHIP
(a) Δmedian≥0.21 PASS (+0.357); (b) Wilcoxon **p=0.45 FAIL** (noise); (c) n/a (2/3 beat,
needs 5 seeds); (d) no gate regression PASS. Gate 0/3.

### CONCLUSION — the FLAT turnover penalty cannot satisfy both aggregate AND consistency
Two data points bracket it: the mechanism only bites at a STRONG penalty (0.045 cuts
cost/gross 111%→69%, net −5.8k→+21.7k) but a strong *flat* penalty over-suppresses ~40% of
folds (fails Wilcoxon, p=0.020); a WEAK penalty (0.022) barely reduces turnover (cost/gross
→107%), collapses the aggregate, and does NOT improve consistency (42:33 vs 44:31) — it even
degrades 2/3 seeds. **The level sweep is concluded — no flat level satisfies both; do not
sweep further (e.g. 0.09 would deepen over-suppression).** Turnover reduction is confirmed as
the first-order lever; the flat, indiscriminate penalty is the limitation. The fix must be
SELECTIVE (tax low-conviction/low-edge entries, not all) or attack the root transfer-failure.

### Decision-log rows (doc 04 §3)
| change | seeds | Δmedian | Wilcoxon | decision | notes |
|---|---|---|---|---|---|
| turnover 0→0.022 | 42,43,44 | +0.357 | p=0.45 (FAIL) | **NO SHIP** | aggregate collapses; consistency NOT improved; degrades 2/3 seeds; concludes the flat-level sweep |

### PR #2 review round 2 — all 4 comments addressed; all 9 threads resolved
- `ab_report --anchor` now auto-resolves running-best from INDEX.md (`43a03ee`); leg(c)
  shown `n/a — needs 5 seeds` in preview; INDEX 5-seed row relabeled (extended + NO-SHIP).
- `run_baseline` fail-fast requires `--candidate` when `turnover_penalty_r!=0` (`0bb4972`) —
  a variant can't silently move running-best; applied post-run; `--job` workers bypass it;
  verified (fires w/o --candidate & creates no dir; workers unaffected; anchor not blocked).

---

## Task 17 — A/B #2: cost-domain randomization — IMPLEMENTED + PINNED + LAUNCHED (2026-07-04)

Reviewer chose A/B #2 (cost-domain randomization) after the turnover-level sweep concluded.
Targets the ROOT of the per-fold inconsistency: fits-train-fails-transfer (train+val− 33%).

**Implemented (committed).** `cost_rand_frac` knob (config default 0.0 = anchor; `COST_RAND_FRAC`
env override). In `env_bracket.reset()` the TRAIN env draws a per-episode multiplier
`m ~ U[1−f, 1+f]` (mean 1.0) that scales `_half_cost`; at f=0 no RNG is drawn (bit-identical
anchor). **TRAIN-ONLY:** `build_env`'s `cost_rand_frac` defaults to 0.0 and only the train
builders (`_spawn_train_env`, n_envs=1 builder) pass `CFG.cost_rand_frac`; every eval/val/test
rollout keeps 0.0 → the equity-based metric is NEVER randomized.

**Isolation PROVEN.** (1) `checks/check_cost_randomization.py`: off-by-default no-op, multiplier
mean-held (0.997≈1.0) in [0.6,1.4], validation rejects <0/≥1/NaN/inf. (2) RULER PROOF: a test
rollout of the stored anchor `f01_s42` is **byte-identical** to the anchor at BOTH
`cost_rand_frac=0.0` AND `0.4` — eval/metric pinned regardless of the training knob. (3) turnover
unit test still green (no regression). Guard + registry extended to `cost_rand_frac`.

**PIN — `cost_rand_frac = 0.4` (U[0.6, 1.4]), pinned BEFORE launch (no result-tuning).**
- *Mean held at 1.0* → the measured cost LEVEL (spread 0.0623 / slip 0.0030) is unchanged; only
  its DISPERSION is learned against. This isolates cost-ROBUSTNESS, not a cost-level change.
- *±40%* = the pre-committed band (doc-11 recommendation). Real XAUUSD spread varies intraday/
  by regime by well more than ±40%, so it's a moderate, plausible regularization strength.
- *Hypothesis:* a policy over-fit to the exact pinned cost is brittle; training against a cost
  band should regularize toward cost-robust (→ regime-robust) policies, raising val-transfer
  (shrinking train+val−) and — the goal — improving per-fold CONSISTENCY (A/B #1's failing leg).
  Secondary: flips are cost-sensitive, so churn may fall (measurable via flip%).
- *Pre-committed reads (no mid-run tuning):* median beats anchor + Wilcoxon consistency + eligibility
  held → extend to 5-seed finalist; no effect (median≈anchor, transfer quadrant unchanged) →
  cost-robustness isn't the lever, move to A/B #3; worse → band too wide, retry 0.2 or conclude.

**Launch:** `COST_RAND_FRAC=0.4 run_baseline.py --label costrand-p40 --seeds 42,43,44
--concurrency 2 --candidate` — 75 jobs, ~5.4h. Same folds/budgets/gate as the anchor; ONLY
`cost_rand_frac` differs. Lockbox sealed. On completion: `ab_report` vs anchor (ship preview) +
transfer diagnostics (train+val− quadrant, flip%). **STOP after the 3-seed report for sign-off.**

---

## Task 18 — A/B #2: cost randomization 0.4 — NO SHIP; clean NULL result  ✅ (2026-07-05)

Run `20260705-072309_c2eb933_costrand-p40`: 75/75 jobs, all `cost_rand_frac=0.4`,
`turnover_penalty_r=0.0` (parity verified), running-best UNCHANGED. **The intervention moved
nothing it was designed to move.**

### vs the anchor (shared 3 seeds)
| metric | anchor | costrand 0.4 | Δ |
|---|---|---|---|
| median-of-3 | −0.5282 | −0.5503 | **−0.024** |
| Wilcoxon p (75 pairs) | — | **0.99** | medianΔ −0.025 |
| improved:regressed | — | **37:38** | coin flip |
| **train+val− quadrant** | **33.0%** | **34.5%** | unchanged |
| train+val+ | 38.6% | 38.9% | unchanged |
| eligibility | 0.386 | 0.389 | unchanged |
| flip % | 35.4% | 34.3% | −1.1 (noise) |
| net cash | −5,782 | −5,708 | +74 (noise) |

Per-seed noise: s42 −0.550, s43 **+0.142**, s44 **−0.782** (one up, one down, one flat).
Ship rule 1/4 (only 'no gate regression'); **SHIP preview = False**.

### CONCLUSION — cost-robustness is NOT the lever; ruled out
±40% mean-held cost randomization left the metric, the **train+val− transfer quadrant**, turnover,
eligibility, and net all statistically unchanged (p=0.99). The dominant failure
(fits-train-fails-transfer, 33%) is therefore **NOT cost-fragility** — plausibly because the cost
is already ATR-relative (regime-invariant by construction), so there was little cost-brittleness
to regularize away. Transfer failure is more likely **representation/feature overfitting** (the
agent fits train-era price/feature patterns). Widening the band is unlikely to help (the
mechanism didn't engage at all, not merely weakly). **Cost randomization is concluded — no ship.**

### Phase-C picture so far (2 A/Bs, both NO SHIP; anchor −0.526 stands)
- **A/B #1 turnover:** the ONLY lever that moved the metric strongly (net −5.8k→+21.7k at 0.045)
  but fails per-fold consistency (flat penalty over-suppresses ~40% of folds); level sweep proved
  no flat level satisfies both.
- **A/B #2 cost-rand:** null — transfer isn't cost-driven.
→ Next should either **salvage the confirmed turnover lever with a conviction-scaled (selective)
penalty**, or attack the transfer overfitting from the **representation/regularization** side
(the cost side is now ruled out).

### Decision-log row
| change | seeds | Δmedian | Wilcoxon | decision | notes |
|---|---|---|---|---|---|
| cost_rand_frac 0→0.4 | 42,43,44 | −0.024 | p=0.99 | **NO SHIP** | null; train+val− 33→34.5% unchanged; cost-robustness not the lever; transfer failure is not cost-driven |

---

## Task 19 — A/B #3: flip-aware turnover penalty — IMPLEMENTED + PINNED + LAUNCHED (2026-07-05)

Reviewer chose the flip-aware (conviction-scaled) turnover penalty to salvage A/B #1's lever.
(The literal action-prob/value-scaled version was ruled out: the env reward can't cleanly see
policy internals, and reward-by-value is circular with GAE — a heavier PPO-rollout change.)

**Implemented (committed).** `turnover_entry_frac` knob (config default 1.0 = A/B #1 flat;
`TURNOVER_ENTRY_FRAC` env override). In `env_bracket.step()` a NEW open carries a penalty weight:
a **FLIP** pays the full `turnover_penalty_r`; a **FRESH** entry pays
`turnover_penalty_r × turnover_entry_frac`. Reward-only, guarded no-op at penalty=0 or weight=0.

**Isolation PROVEN.** `check_turnover_reward.py [4]`: entry_frac 1.0/0.0/0.5 → 4P/1P/2.5P penalty
(the ACTIONS episode = 3 fresh + 1 flip), so **entry_frac=1.0 exactly nests A/B #1's flat 0.045**;
equity/trade-log unchanged (reward-only). Anchor byte-identical invariant re-verified (turnover=0);
cost-randomization unit test still green. run_info + registry record the knob.

**PIN — `turnover_penalty_r = 0.045`, `turnover_entry_frac = 0.0` (tax FLIPS only).**
- *Diagnosis of A/B #1's failure:* the flat 0.045 gave net −5.8k→+21.7k but over-suppressed ~40%
  of folds because it taxed GOOD fresh entries too (failed Wilcoxon p=0.020).
- *This A/B:* keep the validated per-open level 0.045 but charge it ONLY on flips (the 35%
  flip-churn = low-conviction reversals); fresh directional entries pay nothing.
- *Hypothesis:* reduces the reversal churn (preserving A/B #1's net benefit) WITHOUT suppressing
  the high-conviction fresh entries → more UNIFORM per-fold improvement → clears the consistency
  leg A/B #1 failed. Key comparisons: vs anchor (ship) AND vs A/B #1 flat (does sparing fresh
  entries fix consistency?).
- *Pre-committed reads:* consistency improves (Wilcoxon p<0.01) + beats anchor + eligibility held
  → 5-seed finalist; aggregate collapses (net→≤0, i.e. the benefit needed the broad tax) →
  flip-only insufficient, conclude; no change vs A/B #1 → flip-awareness doesn't matter (benefit
  was total turnover reduction, not selectivity).

**Launch:** `TURNOVER_PENALTY_R=0.045 TURNOVER_ENTRY_FRAC=0.0 run_baseline.py
--label turnover-flipaware-p045 --seeds 42,43,44 --concurrency 2 --candidate` — 75 jobs, ~5.4h.
Same folds/budgets/gate; only the turnover mechanism differs (flat→flip-aware). Lockbox sealed.
On completion: `ab_report` vs anchor + consistency/flip% comparison vs A/B #1. **STOP for sign-off.**

---

## Task 20 — A/B #3: flip-aware turnover — NO SHIP; selectivity REFUTED, benefit is BROAD turnover  ✅ (2026-07-05)

Run `20260705-164152_753beea_turnover-flipaware-p045`: 75/75 jobs, all
`turnover_penalty_r=0.045 / turnover_entry_frac=0.0 / cost_rand_frac=0.0` (parity verified),
running-best UNCHANGED. The penalty engaged HARD (flip% 35→8) but the net benefit evaporated.

### Three-way (shared 3 seeds {42,43,44})
| run | median-3 | total trades | flip% | cost/gross | net cash | vs-anchor consistency |
|---|---|---|---|---|---|---|
| anchor | −0.528 | 25,419 | 35.4 | 111% | −5,782 | — |
| **A/B#1 flat 0.045** | **+1.281** | **12,805** | 25.1 | **68%** | **+13,076** | 44:31, p=0.030 |
| **A/B#3 flip-only** | −0.499 | 21,243 | **8.1** | 118% | −6,695 | 39:36, **p=0.751** |

Ship rule 1/4 (Δmedian +0.027, Wilcoxon p=0.75); **SHIP preview = False.**

### CONCLUSION — flip-selectivity REFUTED; the turnover lever is a BROAD cost play
Taxing ONLY flips crushed the flip fraction (35→8%) but the agent **substituted fresh-entry
churn** — total trades fell only 16% (vs the flat penalty's 50%), so cost/gross stayed at 118%
(no better than anchor) and net stayed negative. **A/B #1's +$13k came from halving TOTAL trades
(→ halving cost drag), not from cutting flips specifically**, and that broad reduction cannot be
targeted: spare any trade type and the churn migrates there. Consistency also did not improve
(39:36, p=0.75). This confirms the pre-committed "benefit needed the broad tax" read.

### Phase-C synthesis (3 A/Bs, all NO SHIP; anchor −0.526 stands)
- **#1 turnover flat:** net +$13–28k by halving total turnover (cost drag) — but inconsistent
  (broad suppression hurts ~40% of folds); no flat level fixes it.
- **#2 cost-rand:** null (transfer isn't cost-driven).
- **#3 flip-aware:** refuted — the turnover benefit is *total* cost reduction, entangled with the
  over-suppression; it can't be made selective.
→ The reward/cost side is largely exhausted: reducing turnover just trims COST drag on a
near-zero-edge system (per the trust investigation, no real OOS edge). The net-vs-consistency
tension is fundamental to a cost play. **The remaining unexplored lever is the EDGE itself —
representation/features/regularization (attack the thin signal + train+val− overfitting), not the
cost/turnover side.**

### Decision-log row
| change | seeds | Δmedian | Wilcoxon | decision | notes |
|---|---|---|---|---|---|
| flip-aware (pen 0.045, entry_frac 0) | 42,43,44 | +0.027 | p=0.75 | **NO SHIP** | flip% 35→8 but net −6.7k (fresh-churn substitution); selectivity refuted; benefit needs broad turnover cut |

