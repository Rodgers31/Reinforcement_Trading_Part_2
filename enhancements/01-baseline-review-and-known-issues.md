# Baseline review & known issues

Verified review of the current single-instrument (XAUUSD) bot: what it actually
does, and a prioritized list of bugs, doc-vs-code mismatches, and realism
caveats. All file/line references are repo-root-relative (e.g. `env_bracket.py:297`).

**Verification method.** The pure pandas/numpy core (`data_loader`, `features`,
`leakage_checks`, `evaluate`) and the real `env_bracket.py` (via a minimal
`gymnasium` stub) were **executed** on synthetic data. `train_ppo`, `run_pipeline`,
and the notebooks require `gymnasium`/`stable_baselines3`/`torch`/`plotly`
(not installed locally) and were **traced line-by-line**.

---

## 1. What the bot is

An RL agent that trades gold (XAUUSD) on **H1 decision bars**. Every hour it
chooses a **direction + a stop-loss/take-profit bracket**; fills are simulated on
the underlying **M1** data. It is not a price predictor — it learns a bracket-
placement policy from reward feedback. It does **not** choose position size
(fixed-fractional risk).

### Pipeline (verified data flow)

```
MT4 M1 CSV ──load_mt_ohlcv_csv──► M1 frame (tz-aware, bar-CLOSE indexed)
                                      │ resample_ohlcv (label=right, closed=right)
                                      ▼
                              H1 decision bars
                                      │ prepare_feature_frame (25 causal ATR-normed features)
                                      ▼
        split_train_val_test / make_sliding_folds / make_walk_forward_folds  (200-bar embargo)
                                      ▼
        BracketTradingEnv  (Gymnasium; 1 step = 1 H1 bar; TP/SL filled on M1 in the next hour)
                                      │ VecNormalize + PPO (stable-baselines3)
                                      ▼
        _ConsistencyEvalCallback → best_model.zip + best_model_vecnorm.pkl
                                      ▼
        consistency gate → promote to models/ → final_holdout_eval.py / notebooks
```

### RL formulation (verified)

- **Action:** `MultiDiscrete([3, 3, 4])` = (direction ∈ {flat, long, short},
  SL ∈ {1.0, 1.5, 2.0}×ATR, TP ∈ {1, 1.5, 2, 3}×R). `env_bracket.py:87`
- **Observation:** 25 market features + 6 position-state features = **31-dim Box**.
  `env_bracket.py:90-96`, `env_bracket.py:127-146`
- **Reward (verified exactly):**
  `(Δequity)/risk_budget + mtm_weight·unrealized_R − holding_penalty`.
  `env_bracket.py:297-300`
- **Costs/fills (verified):** entry = `close + dir·(spread/2+slip)`; exit re-pays
  the spread; **if TP and SL fall in the same M1 candle, SL is taken first**
  (pessimistic). `env_bracket.py:156-161`, `env_bracket.py:250-256`
- **Sizing (verified):** `units = (equity·0.5%) / sl_distance`. `env_bracket.py:173-174`
- **Checkpoint selection:** custom `_ConsistencyEvalCallback` keeps the checkpoint
  maximizing `min(q_train, q_val)` where `q = reward − dd_penalty·maxDD%`, only
  when both legs are profitable and place ≥5 trades. `train_ppo.py:142-289`

### Validation (verified)

- **Default entry point** `python train_ppo.py` → `train_sliding_walk_forward()`
  (NOT the block `train_walk_forward()`). Sliding = train 5y → select on 6m val →
  test true-OOS on 6m → slide 6m; ~34 folds; stitches one compounded OOS curve.
  `train_ppo.py:929-1099`
- Block walk-forward and a single chronological split also exist and share the
  same sealed test. `train_ppo.py:774-912`, `data_loader.py:145-298`

### What was verified by execution

| Check | Result |
|---|---|
| Bar-open → bar-close +1min shift | correct |
| H1 resample OHLC aggregation (label/closed=right) | correct |
| Feature count = 25; zero NaN/inf after prep | correct |
| Feature causality (independent truncation probe) | max diff 0.0 |
| Leakage guardrail | passes |
| Drawdown / profit-factor / win-rate math | correct |
| Entry/exit spread, TP fill, **SL-first** when both hit | correct |
| Position sizing `units = risk_cash/sl_dist` | correct |
| Reward decomposition | exact |
| A "1R" TP nets ~0.94R; a "1R" SL loses ~1.06R (round-trip cost) | confirmed |

---

## 2. Bugs (prioritized)

### B1 — The deployment gate doesn't gate. `[High]`
`_finalize_deployment` **always** promotes the final fold's best model into
`models/` and writes `run_info.json` (with a `gate_passed` flag); on failure it
only drops a `NO_DEPLOY.txt` marker. `final_holdout_eval.py`,
`training_diagnostics.py`, and `test_analysis.ipynb` **never read `gate_passed`**,
so a gate-*failed* model is silently revealed on the sealed holdout and reported.
The README, `config.py`, and the `train_walk_forward` docstring all claim
"nothing ships." `train_ppo.py:751-771`, `final_holdout_eval.py:83-108`
- **Fix:** have the holdout/diagnostics refuse to run (or loudly warn) when
  `run_info["gate_passed"]` is false; and reconcile the docs.

### B2 — Model/normalization mismatch in `train_walk_forward` (block scheme). `[Medium]`
It evaluates the **final** model object returned by `train()` while loading the
**best checkpoint's** VecNormalize, so the per-fold summary *and the gate
decision* are computed on a model that isn't the one deployed. The sliding path
(`_load_fold_model`) pairs them correctly. `train_ppo.py:837`, `train_ppo.py:852-857`
- **Fix:** load the best checkpoint (like the sliding path) before re-evaluating.

### B3 — Same vecnorm mismatch in `notebooks/test_analysis.ipynb`. `[Medium]`
The single-split notebook evaluates the BEST-VAL checkpoint using `VECNORM_PATH`
(the *final* model's normalization), not `best_model_vecnorm.pkl`. Since obs stats
drift across training, the headline test metrics use the wrong normalization.
`test_analysis_folds.ipynb` does it correctly — the fix was never backported.
- **Fix:** pair `BEST_VAL_MODEL` with `best_model_vecnorm.pkl`, as the folds notebook does.

### B4 — The learning-curve diagnostic is dead code. `[Low]`
`training_diagnostics.py` section `[1]` reads `eval_logs/evaluations.npz`, which
**nothing ever writes**: SB3's `EvalCallback` is imported but never instantiated,
and the custom callback writes `consistency_evals.csv` instead. It always prints
"No eval logs found." `training_diagnostics.py:30-35`, `train_ppo.py:12`
- **Fix:** point the loader at `consistency_evals.csv`, or drop the section.

### B5 — Duplicate + mojibake print. `[Trivial]`
`train()` prints the post-eval header once correctly, then again with a corrupted
ellipsis (`â€¦`). `train_ppo.py:515-520`

---

## 3. Doc-vs-code mismatches

- **D1** — README points to `notebooks/XAUUSD_RL_Pipeline_Demo.ipynb`, which
  **does not exist**. Actual notebooks: `test_analysis`, `test_analysis_folds`,
  `ensemble_analysis`, `consistency_evals_analysis`.
- **D2** — `config.py:119` says a failed gate renames `run_info.json` →
  `run_info.NO_DEPLOY.json`; the code writes `NO_DEPLOY.txt` and *keeps*
  `run_info.json` (see B1).
- **D3** — The **ensemble portfolio** (`ensemble_analysis.ipynb` blends the last-N
  sliding folds as an equal-capital portfolio of independent books) is the
  intended deployment form but is undocumented in the README/Files list.

---

## 4. Realism / methodology caveats (not bugs)

- **R1 — Realized-only equity.** Equity updates only when a trade closes, so
  **max drawdown understates true intra-trade drawdown**. `env_bracket.py:302-309`
- **R2 — Non-standard Sharpe.** A per-bar Sharpe over a mostly-flat step series,
  annualized by the full H1 bar count (`periods_per_year ≈ 6003`). Noisy/easy to
  misread. `evaluate.py:27-34`
- **R3 — Approximate intrabar fills.** Detection compares **raw mid** M1 High/Low
  against brackets placed off the **spread-adjusted** entry, then re-applies spread
  at the fill — not a true bid/ask intrabar model. Roughly conservative with SL-first.
- **R4 — MTM shaping integrates over time.** The shaping term uses the *level* of
  unrealized R each bar, not the per-bar change, so it mildly rewards holding.
  Small weight (0.01). `env_bracket.py:290-300`
- **R5 — No empty-split guard.** With `embargo=200`, small/demo data silently
  yields empty val/test (reproduced: train/val/test = 24/0/0 on a small set).
- **R6 — Mixed time reference.** `tod_sin/cos` use broker-local hour; session
  flags use UTC hour, so DST shifts time-of-day but not sessions. `features.py:34-52`
- **R7 — Compute cost.** Sliding default ≈ 3M steps/fold × ~34 folds ≈ ~100M
  env-steps. Very expensive.
- **Single-instrument cost model.** One global `spread_price=0.20` in absolute
  price units (`env_bracket.py:157`, `config.py:143`). This is the main blocker for
  multi-instrument work — see doc 02.

---

## 5. Overall assessment

Well-architected research code, not yet a trustworthy production pipeline. Data
handling, causal-feature discipline, embargoed walk-forward, and consistency-based
checkpointing are above average for an RL-trading repo, and the trade engine is
internally consistent and pessimistic. The gaps cluster in the
**reporting/deployment layer** (B1–B4, D1–D3): they don't touch training
correctness, but they affect *which model you ship and what numbers you believe* —
exactly where a real-money system can't afford silent drift.

**Suggested first PR:** B1–B3 together (make deployment/holdout honor
`gate_passed`, and fix the two vecnorm-pairing mismatches).
