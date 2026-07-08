# 08 — Supervised Edge-Ceiling Probe (Task 22)

**Date:** 2026-07-05 · **Branch:** `execution/phase-a` · **Trigger:** reviewer direction after enh/07
(decide *(a) no cost-clearing signal in the features* vs *(b) signal exists but the RL layer can't
extract it* — before committing to the pooling build).
**Protocol:** pinned in `EXECUTION_LOG.md` Task 22 (commit `209179b`) before any label was computed.
**Scope guard honored:** research-only — no RL training, no env/config/feature change, no data
downloads, lockbox untouched (frame truncated at 2024-07-01 before labels existed), anchor and
running-best untouched, no registry/INDEX row.

## Bottom line

**Verdict under the pinned rules: `CEILING-CLEARS` — answer (b), with an era asterisk.**

1. A real, cost-clearing signal EXISTS in the exact 25 features on the exact 25 anchor folds.
   Three of six pre-committed verdict configs clear the measured cost bar in ≥2 of 4 eras;
   shuffled-label controls are clean (median IC +0.0006, AUC 0.497).
2. The signal is **thin and selective**: trading *every* bar is sub-cost even gross
   (all-bar capture ≈ 0.028–0.041 ATR vs the 0.0683 bar); only the **top-|signal|-quintile**
   of bars clears (0.09–0.18 ATR/trade in eras 1–3).
3. It is **regime-decaying at the anchor's 5y train window**: era 4 (2020-07→2024-01, the
   deployment-adjacent era) clears **nowhere** at 5y. The pinned 10y widening arm — neutral
   overall — **rescues era 4 specifically** (top-quintile capture 0.011→0.146 ATR, 6/7 folds
   clear vs 2/7; post-hoc emphasis, flagged below).
4. The RL layer extracts almost none of this. Where the ceiling was highest (E2) the anchor is
   its only profitable era (+7.2% median, PF 1.12); in E1/E3 the ceiling cleared and PPO still
   lost. The bottleneck is **extraction** (unselective trading + memorization), not absence of
   signal.

**Recommendation (§7): do NOT invoke the enh/07 stopping rule (the pinned NULL branch was not
taken). Branch-B comparison: train-window widening (5y→10y) now has direct, free, measured
supervised support and should be RL A/B #4; the pooling path continues per enh/07 but gets a
cheap pooled-supervised-probe go/no-go inserted into Phase 0 before the 1–2-week build.**

---

## 1. Question and design

enh/06 established the anchor (−0.526, gates 0/5) is trustworthy; enh/07 established that
reward/cost shaping is exhausted and the weakness is thin-signal transfer. What no artifact yet
established: whether the 25 features contain ANY signal that clears honest cost — the ceiling on
what *any* extractor (RL included) could do with this observation space.

Supervised probes answer that far more cheaply than RL A/Bs: direct labels, convex/deterministic
fits, per-fold OOS metrics, shuffled-label falsification. If the supervised ceiling is below cost
everywhere, no amount of PPO tuning can win — invoke the stopping rule without the pooling build.

Full pinned protocol: EXECUTION_LOG Task 22 (`209179b`). Summary: anchor-identical data path
(`_load_decision_features()` → `make_sliding_folds`, 25 folds asserted index-identical, embargo
200, lockbox carve), the 25 market feature columns only, fit on TRAIN / report on TEST, labels =
k-bar ATR-normalized forward return (k∈{2,4,8} = anchor holding p25/50/75) + TP-before-SL within
24 bars for the canonical (SL 1.0×ATR, TP 1R) and modal-anchor (SL 2.0×ATR, TP 3R) brackets,
models = Ridge/Logistic + `HistGradientBoosting` (fixed hyperparameters, zero tuning), plus a 10y
widening arm on the 15 folds with 10y history.

## 2. Integrity

| check | result |
|---|---|
| fold boundaries | all 25 re-derived == `make_sliding_folds` output (asserted) |
| shuffled-label control (150 reg / 200 cls fold-fits) | median IC **+0.0006** (pinned gate ±0.02), median AUC **0.4968** (gate 0.48–0.52) — **PASS** |
| per-fold shuffle noise band | IC p5–p95 = [−0.045, +0.053] → single-fold ICs of ±0.05 are noise; all claims below use 25-fold medians/sign-counts |
| NaN metrics | 0 across all 910 real fits (the macOS Accelerate matmul warnings in the log were benign) |
| selection calibration | train-derived thresholds transfer: top-quintile selects 18.6–19.9% of test bars; top-decile 7.2–11.2% |
| runtime | 885 s, single machine, `.venv` sklearn 1.6.1 |

Coverage: first train bar 2006-01-16 → last test bar 2024-01-16 (**18.0y walk-forward coverage**);
stitched OOS = 2011-07-28→2024-01-16 (12.5y, 69,282 H1 test bars; median 2,753/fold, ~29.7k train
bars/fold at 5y). Eras: **E1** f01–06 (tests 2011-07→2014-07), **E2** f07–12 (2014-07→2017-07),
**E3** f13–18 (2017-07→2020-07), **E4** f19–25 (2020-07→2024-01).

## 3. The cost bar (measured, never tuned)

Round-trip cost = `spread_atr_frac + 2×slippage_atr_frac` = **0.0683 × entry-ATR**.

- *k-bar sign strategy:* clears when gross capture per trade ≥ **0.0683 ATR**.
- *Average-bar IC needed* (context): IC_req ≈ 0.0683/(0.7979·σ_k); at k=4, σ per era = 1.45–1.63
  → **IC_req ≈ 0.052–0.059**. Observed all-bar IC is 0.02–0.03 → indiscriminate trading cannot
  clear; concentration on high-conviction bars is structurally required.
- *Brackets:* win rate needed p\* = (SL + 0.0683)/(SL + TP) in ATR units: canonical **0.5342**
  (base ≈ 0.50 → +3.4pp of real hit-rate edge needed); modal **0.2585** (base ≈ 0.085–0.10 within
  H=24 — the 3R target rarely resolves inside a day, so the modal bracket needs ~2.6–3× base-rate
  lift).

## 4. Results

### 4.1 Primary (k=4) — era-median OOS, 5y train

| config | metric | E1 | E2 | E3 | E4 | clears (bar) |
|---|---|---|---|---|---|---|
| fwd4·HGB | top-q5 capture (ATR) | 0.065 | **0.181** | **0.101** | 0.011 | **E2,E3** (E1 misses by 4%) |
| fwd4·ridge | top-q5 capture (ATR) | **0.120** | **0.149** | **0.100** | 0.058 | **E1,E2,E3** |
| fwd4·HGB | OOS IC (median 0.0302, 19/25 folds >0) | 0.043 | 0.058 | 0.022 | **−0.004** | — |
| fwd4·ridge | OOS IC (median 0.0165, 18/25 folds >0) | 0.042 | 0.038 | 0.008 | 0.012 | — |
| canon-long·HGB | AUC / top-decile TP-rate (p\*=0.5342) | .540/.551 | .540/.551 | .531/.550 | .505/.531 | **E1,E2,E3** |
| canon-short·HGB | AUC / top-decile TP-rate | .536/.530 | .540/.564 | .522/.495 | .529/.529 | E2 only |

Fold-level (fwd4): **15/25 folds clear** (ridge), 13/25 (HGB). Per-era clear counts (ridge):
E1 4/6, E2 4/6, E3 4/6, **E4 3/7**; HGB: 2/6, 5/6, 4/6, **2/7**. Trading *all* bars: capture
0.028 (HGB) / 0.041 (ridge) — **sub-cost even gross**, confirming the selectivity requirement.

### 4.2 Secondary configs (diagnostic)

- **fwd8·ridge is the most era-stable config: capture 0.146/0.215/0.202/0.076 — clears ALL 4 eras**
  including E4 (margin +11%). Slower signal survives longer. (Secondary under the pin — noted,
  not verdict-driving; fwd8·HGB is noisy, E1 −0.157.)
- fwd2 mirrors fwd4 with smaller margins; E4 dead (HGB −0.070).
- **Modal bracket paradox:** highest AUCs of the whole probe (logit 0.630 pooled, up to 0.648 in
  E1) yet **never clears** — top-decile TP-rate 0.15–0.24 vs p\*=0.2585. The features rank
  "will a 6×ATR move happen within 24 bars" well (volatility-regime information), but the
  absolute probability stays far below what the wide bracket must pay for. Ranking ≠ clearing —
  and a warning for any "the agent likes 2.0/3R brackets" narrative: its favorite bracket has the
  *least* clearable label.

### 4.3 The supervised transfer gap (mirror of the RL train+val− quadrant)

| model | train IC (median) | OOS IC | gap |
|---|---|---|---|
| HGB | **0.577** | 0.030 | 0.546 |
| ridge | 0.073 | 0.017 | 0.057 |

HGB keeps 3.7–10.3% of its in-sample rank correlation OOS in E1–E3 and **−0.6% in E4**. The
representation supports massive memorization with a sliver of transfer — the same surface PPO
faces through a far noisier gradient. Ridge (57× less in-sample fit) loses far less OOS —
capacity control matters more than capacity on these features.

### 4.4 Where the signal lives (univariate OOS screen, k=4)

Strongest single features are micro-seasonality and candle-shape, not trend/momentum:
`dow_cos` −0.028, `tod_cos` +0.020, `lower_wick_ratio` −0.019, `session_london` −0.017,
`atr_fast_slow` +0.016, `close_ema200_atr` −0.015. In E4-only, the calendar/session/wick trio
persists (−0.026/+0.021/−0.019) while `dow_sin` and `session_asia` die/flip. The multivariate
median (0.030) is only ~1.1× the best univariate — the model mostly aggregates many ~0.02 IC
micro-effects. Fragile by construction; consistent with decay under rising market efficiency.

### 4.5 Widening arm (5y→10y, same val/test windows, 15 folds)

Pinned aggregate: **neutral** — ridge ΔIC median −0.002 (8/15 improved, Wilcoxon p=0.36); HGB
+0.010 (10/15, p=0.17). But the era split (post-hoc emphasis — see §8):

| era (folds) | HGB q5-capture 5y→10y | folds clearing 5y→10y | HGB ΔIC |
|---|---|---|---|
| E2 (2) | 0.234→0.223 | flat | +0.005 |
| E3 (6) | 0.101→0.083 | still clears | −0.007 |
| **E4 (7)** | **0.011→0.146** | **2/7→6/7** | **+0.018 (7/7 folds improve IC)** |

Ridge E4: 0.058→0.155. The E4 signal "death" at 5y is therefore substantially a **training-window
artifact**: models trained on 2015-2020 alone fail on 2020-24, while adding 2010-2015 restores
cost-clearing capture. The signal is not (yet) secularly dead. Cost of this fix: **zero new data**.
(One fold, f24, degrades below the bar at 10y — the rescue is strong but not uniform.)

### 4.6 Ceiling vs what PPO realized (the extraction gap)

| era | supervised ceiling (best q5 capture, 5y) | anchor RL (125 fold-jobs): median ret / PF / % folds >0 |
|---|---|---|
| E1 | 0.120 (1.8× bar) | −1.9% / 0.98 / 33% |
| E2 | 0.181 (2.6× bar) | **+7.2% / 1.12 / 57%** |
| E3 | 0.101 (1.5× bar) | −5.3% / 0.91 / 17% |
| E4 | 0.058 (0.8× bar) | −2.7% / 0.96 / 34% |

PPO is profitable only in the era where the ceiling is ~2.6× cost, and loses in two eras where a
fixed-hyperparameter sklearn model cleared. Mechanism (consistent with every Phase-B/C artifact):
the anchor trades ~39% of bars with 35% flip churn — it monetizes something like the **all-bar**
signal (0.028–0.041, sub-cost) instead of the **top-quintile** signal (0.09–0.18, clearing).
The RL layer needs roughly 2.6× cost of available ceiling before it breaks even — that
inefficiency, not signal absence, is the binding constraint in E1–E3.

## 5. Verdict

**(b) — with an era asterisk.** Signal exists in these 25 features on these folds; it cleared
honest cost in 3 of 4 eras at the anchor's own 5y window (3 verdict configs clear ≥2 eras;
`CEILING-CLEARS` per the pinned rules), and the one dead era (E4) is substantially rescued by a
free train-window change (10y), with the longest-horizon linear config (fwd8·ridge) clearing all
4 eras even at 5y. Option (a) — "nothing in the features clears cost" — is **refuted**. What
remains true and sobering: the signal is thin (~0.02–0.06 IC), selective (top-20% of bars only),
concentrated in micro-seasonality/candle-shape features, decaying at short train windows, and the
RL layer currently captures ~none of it.

## 6. Context correction: doc-02's "limited data" rationale predates the backbone

Doc 02 (§1) argues pooling because "this system's #1 weakness is overfitting a thin signal on
**limited data**, and pooling N instruments is the strongest available regularizer." That was
written 2026-07-01 (`e6bc8ad`) — **one day before** the 2026-07-02 switch to the validated
Dukascopy backbone (`033e779`: start 2006-01-01, lockbox pinned). At writing time the headline
result stood on a ~2.4y bull window of an unvalidated MT5 file. The ratified anchor now stands on
**18.0y of walk-forward coverage (12.5y stitched OOS, 69k test bars)** — "we have little data" is
no longer the load-bearing premise. Pooling's case must rest (and per this probe, can rest) on:

1. **Regularization against memorization** — §4.3's 0.55 transfer gap is exactly the pathology
   cross-instrument data attacks; and
2. **Testing whether the E4/recency decay is gold-specific or universal** — cross-sectional
   evidence no amount of XAUUSD history can provide.

## 7. Branch-B comparison and recommendation

| | **Widen train window 5y→10y** | **Multi-instrument pooling (doc 02)** |
|---|---|---|
| new data needed | none | XAGUSD (+ optional FX major) — not yet acquired |
| supervised evidence | **measured, positive where it matters** (E4 rescue §4.5; overall neutral) | none possible yet |
| cost to test in RL | small: fold-grid extension (probe already validated the windowing: same val/test, t0 −5y), 15 fold-jobs/seed vs 25 | ~1–2 week build after acquisition + Phase-0 |
| main risk | post-hoc era read overstates it; 2× train bars/fold → more to memorize for PPO even if the *feature* signal improves; E3 slightly degrades | pooling could null; Phase-0 may show instruments too dissimilar |
| what a null means | 10y feature-signal ≠ PPO-extractable → extraction gap is the whole problem | stopping rule (enh/07) triggers |

**Recommended sequence:**

1. **RL A/B #4 = 10y sliding train window** (folds 11–25, identical val/test grid, anchor compared
   on the same 15 folds; pinned protocol + ratified ship rule + 3-seed rank → 5-seed finalist,
   unchanged). It is the only lever with *measured* supervised support, costs no data and no
   build-week, and directly targets the deployment-adjacent dead era. This is a data/representation
   lever, not a reward/cost A/B — consistent with enh/07's "do not run more reward/cost A/Bs."
2. **Continue the enh/07 pooling path in parallel** (acquire XAGUSD → Phase-0), with one
   refinement: **extend Phase-0 with this exact probe harness run pooled** (XAUUSD+XAGUSD
   features, same folds/labels/models, instrument tag) — ~1 day once data exists. Go/no-go for
   the RL pooling build: pooled supervised ceiling must beat single-instrument in E3+E4 (the
   decay eras). If it does not, invoke the enh/07 stopping rule **without** paying for the
   build — this moves the stopping-rule trigger earlier at near-zero cost. *(Refinement requires
   reviewer ratification; the stopping rule itself is unchanged.)*
3. **Mandatory reporting, both paths:** every future candidate reports the E4 fold subset
   (f19–25) separately. The probe shows aggregate medians can be carried by 2011-2020 signal
   that no longer exists — a candidate that ships on E1–E3 strength while E4 stays sub-cost is
   fitting a dead signal. *(Reporting requirement only; not a new gate unless ratified.)*
4. **Extraction-gap diagnosis stands regardless of branch:** any build that reaches RL must
   confront §4.6 — PPO trades ~5× too many bars for the signal's selectivity. If A/B #4 and
   pooling both fail the ship rule while the supervised ceiling clears, the honest conclusion is
   an *architecture* negative ("PPO-on-thin-selective-signal doesn't extract"), distinct from
   enh/07's *edge* negative — and the stopping rule still applies as pinned.

## 8. Honesty box

- **Multiple comparisons:** 6 verdict configs; ridge sign-consistency landed exactly on the 18/25
  threshold. Mitigations: shuffle gate passed, capture margins in clearing eras are 1.5–2.6× the
  bar (not threshold-hugging), and the two model families + both label types agree on the era
  pattern. Residual risk acknowledged.
- **The E4-widening read is post-hoc.** The widening arm was pinned; its *era decomposition* was
  not a pre-committed verdict rule. All-fold Wilcoxon is p=0.17 (n.s. at n=15). The 7/7 IC
  improvement and 2/7→6/7 clearing in E4 is strong but should be confirmed (e.g., a cheap 8y/12y
  supervised variant) before heavy investment — A/B #4 *is* that confirmation at the RL level.
- **A ceiling, not a strategy.** Top-quintile capture is gross, per-trade, H1-touch-scanned with
  the env's pessimistic same-bar rule, no DD/gate dimension, no slippage beyond the pinned model,
  overlapping-entry serial correlation left in (fold-level aggregation is the honest unit). It
  bounds what the feature set offers; it does not promise a deployable system.
- **Lockbox:** starts 2024-07-01 — *after* E4. At 5y windows E4 was already sub-cost; the 10y
  rescue is measured only through 2024-01. Whether it persists into the lockbox era is exactly
  what must never be peeked at.

## 9. Files

- `enhancements/08_probe/probe_edge_ceiling.py` — probe (pre-committed `a26727e` before launch)
- `enhancements/08_probe/analyze_results.py` — post-run tables (this doc's numbers)
- `enhancements/08_probe/results_per_fold.csv` — 910 real + 350 shuffle fit rows
- `enhancements/08_probe/results_univariate.csv`, `fold_windows.csv`, `summary.json` (pinned-rule
  verdict evaluation), `probe_stdout.log` (trimmed of benign BLAS warnings)
- Protocol pin: `EXECUTION_LOG.md` Task 22 (`209179b`); results appended same task.

*No training was run; the anchor stands; running-best is unchanged.*
