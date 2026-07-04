# 06 — Baseline Trust Investigation

**Date:** 2026-07-03 · **Branch:** `execution/phase-a` · **Anchor:** `runs/20260703-021710_b9bc9d6_baseline-3seed/` (5-seed set, ratified `6e536a2`)

**Mandate.** Settle one question with proof, not assertion: *why does the old notebook show +70% while the honest baseline shows −0.53 gate-FAIL, and can EITHER result be trusted?* Four agents were pointed at four falsifiable questions with an **adversarial** brief — try to **disprove** the team's working conclusion, not confirm it. Rules: read-heavy; no model trained or evaluated; nothing touched sealed data (≥ 2024-07-01); every claim cites a primary source (git ref, on-disk artifact, data file, or `file:line`). The orchestrator then independently reproduced every decision-critical number before writing this report (see **Verification log**).

---

## Bottom line

The old **+70% cannot be trusted** and the honest **−0.526 can** (with one caveat about *why* it loses). The +70% is fully absorbed by **[bull-market beta] + [single lucky window] + [cost/fill flattery]**, with a **B3 normalization** residual whose sign is unknowable — leaving **no demonstrable residual of real net edge**. The honest baseline survives a money-losing-bug audit, a fair-cost audit (the measured recent spread is *higher* than what the model was charged), an under-training audit (the jobs are *over*-trained), and a rigged-ruler audit (a passive control makes money in bull windows and loses in bear windows through the *identical* harness). The seed-luck band (0.63) and window-luck band (7.35) are wide enough that a single-seed/single-window number like +70% is one draw from the *same* honest pipeline that medians to −0.526.

Two of the four attacks landed a hit worth recording, and neither disproves the conclusion:
1. **The agent underperformed buy-and-hold.** Over the old test window gold returned **+124.05%**; the agent captured **+70%** = **0.56× beta**. The "win" was a *sub-beta* long proxy.
2. **A real but small GROSS edge exists in the honest baseline** — at *zero* cost 2/5 seeds pass the gate — but a *fairly-measured* spread consumes it entirely. This changes the *narrative* (loss is cost-driven, not signal-absent), not the *verdict* (0/5 under honest cost). It does **not** halt Phase C.

---

## Verdict

**OLD +70% — NOT TRUSTWORTHY.** One seed (42) × one `split_train_val_test` window (2024-02-06 → 2026-05-30), scored under a normalization mismatch, on an env whose fixed $0.24 round-trip cost is trivial against an $8–12 ATR. Buy-and-hold over that same window returned +124.05%, so beta alone more than accounts for the gross; the thin PF (1.144) is erased when the same trades are re-charged at the honest ATR-scaled cost (PF → 1.01–1.09). No residual attributable to skill survives.

**NEW −0.526 (gates 0/5) — TRUSTWORTHY WITH CAVEATS.**

*Proven:*
- **Accounting is clean.** Cost charged once per side (`env_bracket.py:164-176`), round-trip `= (spread_atr_frac + 2·slippage_atr_frac)·ATR = 0.0683·ATR`; `sum(pnl) == final_equity − 10000` exactly (f01_s42); MTM curve equals equity when flat (no drawdown inflation); stitching is correct multiplicative compounding. *(Agent B; cost code re-read by orchestrator.)*
- **Cost is fair, arguably lenient.** The only directly measurable ask−bid spread fraction (2023, `data/dukascopy_raw/xauusd-m1-{bid,ask}-2023.csv`, n=352,969) is **0.083–0.090 ATR-frac — higher than the charged 0.0623** (`config.py:184`). Halving cost has no empirical basis. *(Agent B measured 0.0825; orchestrator reproduced 0.089–0.090.)*
- **Jobs are over-trained, not under-trained.** Only 3/125 (2%) peak in the final 10% of training; `train_eval_r` rises to 21.2 while `val_r` plateaus at ~5–7 (`eval_logs/consistency_evals.csv`). More steps → worse OOS. **No retrain warranted.** *(Agent B.)*
- **The ruler is fair, not rigged-pessimistic.** A passive long-only control through the *identical* harness+cost is positive in 5/5 strong-bull folds and negative in 3/3 strong-bear folds; corr(long-only fold return, gold B&H) = 0.896; sign agreement 22/25. A rigged harness cannot produce this. *(Agent C.)*
- **A single number is one draw from a wide luck distribution.** Seed-only band 0.626 (−0.616 … +0.0095) at fixed data/architecture; window band across 125 fold-runs **7.345** (−0.9629 fold13/s43 … +6.3821 fold9/s45). The honest median over 25×5 draws (−0.526, IQR 0.419) averages away both luck axes. *(Agent D; min/max/fold-25 re-verified by orchestrator.)*
- **The pipeline is alive.** Fold 25 (2023-H2) is positive on all 5 seeds (mean +9.02%, PF 1.214); fold 9 (2015-H2) positive on all 5 (+15.99% … +34.61%); 14/25 folds have unanimous return-sign. It prints profit where edge exists — which is exactly what makes the old +70% explainable as a favorable single draw. *(Agent D; re-verified.)*

*Open (does not change the verdict):* see [Open questions](#open-questions).

---

## The gap, decomposed

Old **+70%** (doc-confirmed: one seed × one split, fixed $0.24 cost, normalization-mismatched) → honest **−0.526** (25 folds × 5 seeds, ATR-scaled cost). Sized, sourced components:

| # | Component | Size / direction | Evidence |
|---|---|---|---|
| 1 | **Bull-market beta** | Window buy-and-hold = **+124.05%**; agent captured **+70%** = **0.56× beta** — a *worse-than-long-only* proxy. Beta more than fully accounts for the gross return. | `XAUUSD_M1` Close, **verified** (first 2025.23 → last 4537.50) |
| 2 | **Cost / fill flattery** | Old env: fixed **0.24 price** round-trip (`spread_price 0.20`+`slippage 0.02`), ≈ **0.010–0.029 R/trade** against ATR 8–12. Re-charging the **same 1487 trades** at the honest **0.0683·ATR** cuts net from **111.9 R → 9–76 R**, PF **1.144 → 1.01–1.09** ⇒ **18–51 pts** of the return. | `git 7615a58:env_bracket.py:54-55,157` (old) vs `env_bracket.py:164-176` (honest); Agent A recompute on stored trade stats |
| 3 | **Single-window luck** | ONE split vs median of 25 contiguous OOS windows. Window band = **7.345** metric-units; best single window fold9/s45 = **+34.61%** return / **+6.382** metric. +70% is one draw from this band. | `baseline_report.json fold_metrics_per_seed`; `seed_45_summary.csv` fold 9 — **verified** |
| 4 | **Seed luck** | ~**0.63** metric-units; seed band −0.616 … +0.0095, IQR 0.419; sign of stitched return flips at seed 46. | `baseline_report.json metric_per_seed / metric_seed_iqr` — **verified** |
| 5 | **B3 mis-normalization** | Best-val checkpoint scored with the **final** model's vecnorm. **Real but sign/magnitude UNPROVABLE** — the `F:\…best_model.zip` is gone and inference is prohibited. | `git 7615a58:notebooks/test_analysis.ipynb` cell `b8bbe8e7`; fix `e980627`; `checks/check_b2_vecnorm_pairing.py` |
| 6 | **Residual real NET edge (old +70%)** | **None demonstrable.** Beta (124%) > gross (70%); PF → ~1.0 under honest cost; walk-forward median −0.526. | Agents A + B + D |

The gap is **overwhelmingly bull-beta + window/seed luck (1, 3, 4), with cost flattery (2) erasing the thin PF**; B3 (5) adds noise of unknown sign; there is no skill residual (6). *(Distinct from this: the honest baseline itself has a small positive **gross** edge — see RF-1 — which is a fact about the −0.526, not about the +70%.)*

---

## Red flags

**One survived, and it is a nuance, not a disproof. Nothing here halts Phase C.**

**RF-1 (Agent B) — the loss is cost-driven, not signal-absent.** Gross PnL is positive on all 5 anchor seeds; cost consumes **94–125% of gross** (`jobs/*/test_trades.csv`). First-order gate recompute on the stored trades: at **zero** cost **2/5 seeds pass** (s44, s46) and at **half** cost all 5 turn trade-Sharpe-positive. So there is a genuine small **gross** directional edge. **Why it does not rescue the strategy or reopen the ruler:** the applied cost is empirically *fair-to-lenient* — the only directly measurable spread fraction (2023 = **0.083–0.090**) is **higher** than the charged 0.0623, so the honest baseline may be slightly *worse* than −0.526, not better. RF-1 is a **first-order bound** (path-dependence not re-simulated — prohibited), a statement about the *mechanism* of the loss, not a money-losing bug, not an unfair cost, not a rigged ruler.

**No money-losing-bug red flag survived** — accounting exact, no double-charge, no MTM inflation; the 200-bar embargo is a required EMA-anti-leak purge applied identically to every fold, not an outcome-selective filter.

**No rigged-ruler red flag survived** — the passive long-only control wins 5/5 strong-bull folds and loses 3/3 strong-bear folds through the identical harness+cost (corr 0.896 with gold B&H).

---

## A — Decomposing the old +70%

**Falsifiable question:** can the +70% be fully accounted for by beta + single-window luck + cost/fill flattery + B3, leaving no residual of real net edge? **Answer: yes — no residual survives.**

### 1. What the +70% actually is (stored outputs of `git show 7615a58:notebooks/test_analysis.ipynb`)

| Item | Value | Primary source |
|---|---|---|
| Method | **single** split via `data_loader.split_train_val_test(train_frac=0.8, val_frac=0.1, embargo=200)` — NOT walk-forward | cell `1e4b075a` |
| Checkpoint | best-val `best_model.zip` promoted as the primary downstream result | cells `853a550e`, `b8bbe8e7` |
| Test window | **2024-02-06 → 2026-05-30** (13,674 H1 bars) | cell `1e4b075a` |
| Headline return | **+70.0%** literal; n=**1487**, PF **1.144**, Sharpe **1.552**, MaxDD **−10.30%**, WinRate **42.2%**, AvgR **0.075**, net sum-R **111.9** | cells `b8bbe8e7`, `e834bebb`, `68298b99` |
| `gate_passed` | **False** (analyzed silently — the B1 bug) | cell `853a550e` — **verified** |
| Model artifact | `F:\CodeTrading\Code_trading_2026\34_RL_XAUUSD\models\best_model\best_model.zip` — **unrecoverable** (no such `.zip` anywhere in repo) | cell `853a550e`; `find . -name '*.zip'` |
| slug / seed | `ppo_H1_sl1-1.5-2_tp1-1.5-2-3_3000k_seed42`, seed 42 | cell `853a550e` |

### 2. B3 mis-normalization (real, direction UNPROVABLE)

`run_info` carries a separate `best_model_vecnorm_path`, yet cell `b8bbe8e7` scores the best-val checkpoint with the **final** model's `VECNORM_PATH` (`= run_info['vecnorm_path']`). Observation stats drift across training, so the earlier checkpoint was normalized by the wrong statistics. **Direction and magnitude are unprovable** without loading the `F:\…best_model.zip` (gone) and running inference (prohibited). The repo ships `checks/check_b2_vecnorm_pairing.py` for exactly this bug class; the notebook fix is commit `e980627`.

### 3. Cost / fill flattery — SIZED (arithmetic on stored trade stats, no model run)

Old env charged **absolute** cost: round-trip `= 2·(spread_price/2 + slippage_price) = 2·(0.10 + 0.02) = 0.24` price (`git 7615a58:env_bracket.py:54-55,157,161`), i.e. cost-in-R `= 0.24 / (sl_mult·ATR)`. With test-window ATR mean **≈11–12** / median **≈8**, that is **~0.010–0.029 R/trade** — a fixed $0.24 is nothing against an $8–12 ATR on $2000–4500 gold. **That is the flattery.** The honest env charges **ATR-scaled** cost (`env_bracket.py:164-176`): round-trip `0.0683·ATR` ⇒ cost-in-R `= 0.0683 / sl_mult`, *ATR cancels* → fixed ~0.034–0.068 R/trade (higher still at Agent B's measured 0.083–0.090 spread).

Re-charging the **same 1487 trades** (net 111.9 R) at the honest cost:

| avg sl_mult | spread model | extra cost | net R (honest) | PF (honest) |
|---|---|---|---|---|
| 1.0 | cfg 0.0683 | 72.7 R | 39.2 | 1.046 |
| 1.0 | measured 0.089 | 102.8 R | **9.1** | **1.010** |
| 1.5 | cfg 0.0683 | 48.5 R | 63.4 | 1.077 |
| 2.0 | cfg 0.0683 | 36.4 R | 75.5 | 1.093 |

Cost flattery = **36–103 R ≈ 18–51 pts** of the ~56% simple return. The agent trades often at tight stops (sl≈1.0), so the large-extra-cost rows are the relevant ones: honest net collapses toward **~9 R (PF 1.01)** — indistinguishable from zero edge.

### 4. Bull-market beta — VERIFIED

Gold buy-and-hold over the **exact** old test window (2024-02-06 → 2026-05-30), from the backbone CSV: **+124.05%** ($2025.23 → $4537.50). The agent's **+70% is 0.56× the passive B&H** — it **underperformed simply holding gold**. The passive TrendHold baseline +33.75% (cell `e834bebb`, n=505) is *doc-asserted* (not re-run on sealed bars); immaterial, since B&H already dwarfs +70%.

### 5. Line-item decomposition

| Component | Size | Source |
|---|---|---|
| Bull-market beta | Window B&H **+124.05%**; agent ≈ **0.56× beta** | data CSV (verified) |
| Single-window luck | ONE seed42/ONE split; honest 25×5 median **−0.526**, gates 0/5 | anchor; Agent D |
| Cost/fill flattery | **+18–51 pts**; honest re-charge net R 111.9 → **9–76**, PF 1.144 → **1.01–1.09** | env_bracket old vs new; Agent B spread |
| B3 / unprovable residual | sign & size **UNKNOWN** (dead F:\ model) | cell `b8bbe8e7`; §1 |
| **Real net-edge residual** | **none demonstrable** | A+B+D |

**No red flags** — every attempt to isolate skill collapses: beta > gross return, PF → ~1.0 under honest cost, walk-forward median negative.

---

## B — Is −0.53 artificially pessimistic? (Attack the new baseline)

**Falsifiable question:** would a correct / fair / converged run clear the gate? **Answer: No** — −0.53 survives all four attack vectors. Anchor: `baseline_report.json` — metric_median **−0.526**, seeds ∈ {−0.616, −0.528, −0.526, −0.109, +0.009}, gates **0/5**.

### 1. Money-losing bugs — CLEAN
| Check | Result | Source |
|---|---|---|
| Cost double-charged? | **No.** Charged once per side; round-trip `= 0.0623 + 2·0.003 = 0.0683 ATR`. | `env_bracket.py:164-176` |
| PnL sign error? | **No.** `corr(pnl_sign, (exit−entry)·dir) = 1.000` (f01_s42). | `test_trades.csv` |
| Accounting integrity | **Exact.** `sum(pnl)=719.21 == final_equity−10000` (f01_s42). | `test_trades.csv`/`test_equity.csv` |
| MTM inflates the −63% DD? | **No.** `equity_mtm == equity` when flat; the DD is a real gradual bleed. Metric = return ÷ \|max MTM DD\| (`eval_harness.py:28-44`). | `seed_42_stitched.csv` |
| Stitching | Correct multiplicative compounding, no leak. | `run_baseline.py:67-86` |
| Embargo eats winners? | **No bias.** 200 H1 bars purged at the *start* of every fold identically — a required EMA-200 anti-leak purge, not outcome-selective. | `data_loader.py:303` |

### 2. Cost fairness — FAIR, arguably LENIENT
Directly measured 2023 M1 ask−bid spread (market fact, pre-lockbox), `xauusd-m1-{bid,ask}-2023.csv` (n=352,969):

| Quantity | Config asserts | Measured 2023 | Source |
|---|---|---|---|
| Median spread (price) | 0.41 | **0.330** | raw ask−bid |
| Median H1 ATR(14) | 6.586 (2023-26 blend) | **3.65–4.00** (2023) | bid resample |
| **Spread ÷ ATR fraction** | **0.0623** | **0.083–0.090** | ratio |

The config's 0.0623 divides a recent-era spread by an ATR window inflated by the 2024-26 rally. The only *directly measurable* fraction (2023) is **higher** than what the model paid → **halving cost has no empirical basis.** First-order gate recompute on stored trades:

| Cost × | seeds passing | mean trade-Sharpe |
|---|---|---|
| **1.0 (as-run, fair)** | **0/5** | −0.30 … +0.03 |
| 0.5 | 0/5 | +0.28 … +0.62 |
| 0.0 (impossible) | **2/5** (s44, s46) | +0.85 … +1.20 |

Gross PnL positive on all 5 seeds; cost consumes **94–125% of gross**. Genuine gross edge, fairly-charged spread eats it. −0.53 SURVIVES.

### 3. Undertraining — REFUTED (it's OVER-trained)
Only **3/125 (2%)** jobs peak in the final 10% of training; median peak at 47%; `train_eval_r` → 21.2 while `val_r` plateaus at ~5–7 after 1.8M steps. Widening train/val gap = overfitting. More steps → worse OOS. No retrain warranted.

### 4. Checkpoint-fallback drag — REAL but tiny
19/125 jobs (15%) fall back to the final checkpoint. Fallback folds do average worse (fold_metric −0.016 vs +0.334), but dropping them lifts the mean only **0.281 → 0.334** — nowhere near flipping a seed. The negativity is dominated by *fully-eligible* late folds (2017–2023).

**Bottom line:** −0.53 is not a bug, not an unfair cost, not undertraining — the honest verdict of a system whose small positive gross edge a fair spread consumes.

---

## C — Is the ruler fair? (Passive references through the honest harness, no training)

**Falsifiable question:** does the honest harness make even a *known-good* passive strategy lose everywhere, including bull windows (→ rigged-pessimistic)? **Answer: No — the ruler is fair.**

**Method (read-only, mirrors production).** Scratch `passive_ruler_check.py`: features via `train_ppo._load_decision_features()`; folds via `data_loader.make_sliding_folds(train_years=5, val_months=6, test_months=6, step_months=6, embargo_bars=200, lockbox_start='2024-07-01')` → **25 folds**, identical to the anchor. Each fold's TEST window run through `BracketTradingEnv` with the **same** cost (`spread_atr_frac=0.0623`, `slippage_atr_frac=0.0030`) via `baselines.run_policy`; stitched metric = `metric_return_over_mtm_dd` (`eval_harness.py:28`). Lockbox honored (last bar fold 25 = 2024-01-16). No model trained, no NN inference.

| Check | Result |
|---|---|
| Long-only positive in strong-bull folds (gold B&H ≥ +8%: folds 10,16,17,18,23) | **5/5 positive** |
| Long-only negative in strong-bear folds (gold B&H ≤ −8%: folds 4,8,11) | **3/3 negative** |
| Sign agreement long-only vs raw gold direction | **22/25 folds** |
| corr(long-only fold return, gold B&H) | **0.896** |

Long-only captures a large share of each up-move despite the spread: fold 10 gold +19.8% → +12.7% (PF 1.42); fold 18 +15.2% → +9.6%; fold 16 +8.4% → +10.3%. A rigged harness cannot produce this.

**Stitched-metric comparison on the same ruler:**

| policy | stitched metric | return% | PF | folds+ |
|---|---|---|---|---|
| **RL anchor (median of 5)** | **−0.526** | −33.5 | 0.98 | 8/25 |
| passive_ema_atr | −0.885 | −66.4 | 0.94 | 7/25 |
| long_only (≈ buy-and-hold) | −0.551 | −29.8 | 0.97 | 11/25 |
| random | −1.000 | −99.9 | 0.82 | 0/25 |

RL (−0.526) is **not worse than dumb passive** — better than rule-based EMA (−0.885), on par with long-only (−0.551), far above random. The whippy `passive_ema_atr` loses across regimes because it churns ~300 trades/fold paying spread each time (a property of the *strategy*, not the ruler). **Verdict: FAIR ruler.** The −0.53 reflects a genuine lack of OOS edge, not a broken measuring stick.

*(Mandate label fix: fold 9 was pre-labeled "bull," but its raw gold B&H is ≈ **−0.66%** — a chop window — so passive losing there is correct ruler behavior.)*

---

## D — Why walk-forward + multi-seed is the correct measure

**Falsifiable question:** do the 5-seed artifacts prove a single-model/single-window number is luck-dominated, so the old/new gap is window-selection, not a broken pipeline? **Answer: yes — the spread is enormous and the pipeline is verifiably alive.**

### 1. Seed-luck band (seed only differs)
| seed | metric | return% | PF | folds+/25 |
|---|---|---|---|---|
| 42 | −0.5282 | −33.50 | 0.984 | 8 |
| 43 | −0.6160 | −38.71 | 0.979 | 8 |
| 44 | −0.1094 | −4.78 | 1.004 | 9 |
| 45 | −0.5260 | −33.48 | 0.987 | 9 |
| 46 | **+0.0095** | +0.51 | 1.006 | 10 |

Band **0.626** (−0.616 … +0.0095), median −0.526, IQR 0.419. Only the RNG seed moved, yet the metric swings 0.63 and the sign of stitched return flips. A single-seed run cannot be trusted to ±0.3.

### 2. Window-luck band ≈ 12× wider
Pooling 125 fold-runs: fold_metric min **−0.9629** (fold13/s43) … max **+6.3821** (fold9/s45), range **7.345**; test_return min **−17.95%** (fold17/s42) … max **+34.61%** (fold9/s45). Cherry-picking one window yields anything from −18% to +34.6%. This is the mechanical proof the old single-split +70% was a *window* draw — the same honest pipeline that medians to −0.526 prints +34.6% on fold 9.

### 3. The pipeline prints profit on favorable windows
Fold 25 (2023-07-27 → 2024-01-16), per seed: returns +8.57 / +10.30 / +7.05 / +10.51 / +8.68 %, **all 5 positive**, mean **+9.02%**, PF 1.15–1.27. Fold 9 stronger (+15.99 … +34.61%, all 5 positive). **14/25 folds have unanimous return-sign** — genuine window-level signal (which averages to a loss), not a dead reporting channel.

### 4. The extremes are luck, not beta (market facts, verified)
| window | biggest single result | gold buy-and-hold |
|---|---|---|
| fold9 2015-07-29→2016-01-15 | **+34.61%** (best) | **−0.66%** (flat) |
| fold17 2019-07-29→2020-01-16 | **−17.95%** (worst) | **+9.29%** (up) |
| fold25 2023-07-27→2024-01-16 | +9.02% mean | **+2.84%** |

The biggest win landed where gold went nowhere; the biggest loss landed in a rising market. Extreme single-window outcomes are seed×window luck, not directional beta.

### 5. Provenance
`git show 7615a58:notebooks/test_analysis.ipynb`: one model (`…seed42.zip`/`best_model.zip`) on one `split_train_val_test` split (`data_loader.py:145-172`) — one point from a distribution 0.63 (seed) × 7.35 (window) wide. **The honest median over 25 contiguous OOS windows × 5 seeds = −0.526 (IQR 0.419)** averages away both luck axes; the single-split number averages away neither. No subset of the artifacts had a spread tight enough to rehabilitate a single number.

---

## Open questions

1. **Path-dependent cost re-simulation is prohibited.** RF-1's "half cost → 0/5; zero cost → 2/5" and Agent A's honest-cost net-R band are **first-order** (cost added back to realized trades) — a trade that hit SL under full cost might not under half cost. Faithful re-costing needs a trained-model re-run (not permitted). These are directional bounds.
2. **Pre-2023 spread is unmeasurable** — Dukascopy ask data exists only 2023+. ATR-relative scaling makes cost regime-invariant *by construction* but cannot be empirically verified for the 2011–2022 folds.
3. **B3 direction/magnitude is unprovable** — the best-val checkpoint's normalization mismatch cannot be resolved without the dead `F:\` model + inference.
4. **The exact old net figure under honest cost is a band (9–76 R)**, not a point — the per-trade `sl_atr_mult` distribution was never saved outside the `F:\` session.
5. **Slippage 0.0030 is assumed, not measured** (unobservable from OHLC). Minor term vs the 0.0623 spread.
6. **The surviving gross edge (RF-1) is not exploited** — whether any execution regime (tighter real spread, different instrument, netting) realizes it net-positive is out of scope and would need a prohibited re-run.

---

## Training gate

**NONE.** Agent B's under-training check was *conclusive*, not inconclusive: the jobs are **over-trained** (3/125 peak in the final 10%; train/val gap widens to 3M steps), so the scoped "retrain one non-sealed fold at 6M steps" contingency is **not** triggered. No agent proposed a training run requiring approval. **No training is requested.**

---

## Appendix: evidence index

**Anchor** (`runs/20260703-021710_b9bc9d6_baseline-3seed/`): `baseline_report.json` (metric_median −0.526; metric_per_seed {42:−0.5282, 43:−0.616, 44:−0.1094, 45:−0.526, 46:+0.0095}; IQR 0.4188; n_folds 25; gates 0/5; `fold_metrics_per_seed` 125 pts, min −0.9629 / max +6.3821); `seed_{42..46}_summary.csv`; `seed_{n}_stitched.csv`; `jobs/f{NN}_s{seed}/{test_trades,test_equity,run_info,eval_logs/consistency_evals}`.

**Code:** `env_bracket.py:164-176` (per-side ATR cost, round-trip 0.0683·ATR); `git 7615a58:env_bracket.py:54-55,157,161` (old fixed 0.24 cost); `eval_harness.py:28-44` (metric = return ÷ |max MTM DD|); `data_loader.py:145-172,232,303` (split_train_val_test / make_sliding_folds / 200-bar embargo); `config.py:122` (lockbox 2024-07-01), `:184-185` (spread 0.0623, slippage 0.0030); `baselines.py:16,33,62,76`; `run_baseline.py:67-86` (stitching); `checks/check_b2_vecnorm_pairing.py` (B3 bug class).

**Data / market facts:** `data/XAUUSD_M1_Bid_Dukascopy_2003.05.05_2026.07.02.csv` (spans 2003-05-05 → 2026-07-02) — verified B&H: old window 2024-02-06→2026-05-30 **+124.05%**; fold9 **−0.66%**, fold17 **+9.29%**, fold25 **+2.84%**. `data/dukascopy_raw/xauusd-m1-{bid,ask}-2023.csv` (n=352,969) — median spread 0.330, H1 ATR(14) 3.65–4.00 → frac **0.083–0.090** > charged 0.0623.

**Git refs:** `7615a58:notebooks/test_analysis.ipynb` (old +70%, seed42, single split, gate_passed False, model on `F:\`); `e980627` (B3 fix); `b9bc9d6` (anchor); `6e536a2` (ratified 5-seed).

**Scratch scripts (read-only, session scratchpad):** `passive_ruler_check.py` + `passive_*_perfold.csv` + `passive_summary.json` (Agent C); `verify_claims.py`, `verify_A.py` (orchestrator verification).

### What the team tried to break and could NOT
1. That −0.53 is a money-losing bug — audit clean. 2. That the cost is unfairly high — measured 2023 spread is *higher* than charged. 3. That the jobs are under-trained — they are over-trained. 4. That the ruler is rigged-pessimistic — a passive control wins in bull folds, loses in bear folds through the identical harness. 5. That the seed/window spread is tight enough to trust a single number — it is 0.63 / 7.35 wide. 6. That the +70% hides real skill — beta (124%) exceeds the entire gross and honest cost erases the PF. **Every attack reinforced the conclusion.** The sole surviving caveat (RF-1: a small gross edge eaten by fair cost) is about the *mechanism* of the loss, not a reason to distrust −0.526.

### Verification log (orchestrator, independent of the agents)
- **Anchor JSON** read directly: median −0.526, gates 0/5, per-seed & per-fold spreads — confirmed.
- **Cost code** `env_bracket.py:164-176` re-read: charged once per side, round-trip **0.0683·ATR** (corrected Agent B's "0.0685" rounding).
- **2023 spread fraction** reproduced from raw bid/ask: median spread 0.330, ATR(14) 3.65–3.71 → **0.089–0.090** (> Agent B's 0.0825; both > 0.0623). Claim holds, if anything understated.
- **Buy-and-hold** reproduced: old window **+124.05%** (exact match to Agent A); fold9 **−0.66%**, fold17 **+9.29%**, fold25 **+2.84%**. ⚠️ These correct the *synthesis draft*, which had drifted to −1.69 / +9.59 / +4.17 and mislabeled them "verified"; the values above match Agent D's original section and the raw data.
- **seed45 fold9 = +34.614589%** / metric 6.382, pooled min −0.9629 (fold13/s43), **fold 25 all 5 seeds positive** (mean +9.02%) — confirmed from `seed_*_summary.csv`.
- **Old absolute cost** `git 7615a58:env_bracket.py`: `spread_price=0.20`, `slippage_price=0.02` → round-trip 0.24 — confirmed.
