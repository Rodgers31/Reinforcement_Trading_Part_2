# 09 — Supervised Baseline Through the Honest Ruler (Task 23)

**Date:** 2026-07-05 · **Branch:** `execution/phase-a` · **Trigger:** reviewer direction after
ratifying enh/08 §7.2 (pooled-probe Phase-0 gate) and mandatory E4-subset reporting: before A/B #4,
convert the Task-22 ceiling into the anchor's own units and set the benchmark any RL candidate
must beat. **Protocol:** pinned in EXECUTION_LOG Task 23 (commit `6f6db2a`) before any simulation
ran; exactly two pre-declared variants, two train-window arms, zero parameter search.
**Scope guard honored:** research-only — no RL training, no env/config change, no data downloads,
anchor/running-best untouched, no registry/INDEX row, lockbox untouched.
**Status: STOPPED for review. A/B #4 has NOT been launched.**

## Bottom line

**Yes — selective supervised trading clears the ruler where PPO couldn't, but only in one of the
four pinned cells: the 10y train window with time-exit execution.** Three findings, each
load-bearing for what happens next:

1. **V1·10y (folds 11–25): metric +4.26, return +126.2%, PF 1.11 — vs the anchor's −0.78 on the
   identical folds.** Its strongest segment is E4 (+5.43 metric, +80.3%) — the era that killed
   every 5y config. The enh/08 widening rescue survives honest costs, compounding, and MTM DD.
2. **Execution geometry decides everything: the same signal under canonical-bracket execution
   (V2) is destroyed** — −85.3% at 5y, −43.1% at 10y, cost/gross 532%/157%. The signal is 4-bar
   *drift*; ±1×ATR first-touch brackets convert it into a cost-paying coin flip (49–50% win rate
   vs the 53.4% the bracket needs). This is a structural warning for the env's bracket-mandatory
   action space.
3. **Nothing is deployable:** all four cells FAIL the ratified gate. V1·10y fails on exactly one
   leg — the 10th-percentile fold PF floor (0.82 vs 0.90; driven by f24 0.80, f16 0.81, f18 0.84).
   It is a benchmark, not a product.

## 1. What ran, and why you can trust the numbers

- Ruler = the anchor's own unmodified functions: per-fold `evaluate.full_report` →
  `run_baseline._stitch` → `eval_harness.metric_return_over_mtm_dd` →
  `train_ppo._passes_consistency_gate`.
- **Cross-check:** re-stitching the anchor's own 125 `test_equity.csv` files through this exact
  chain reproduced the ratified per-seed metrics (full-25 median **−0.5260**, asserted to <5e-4
  per seed) — the reuse is proven, not assumed. Anchor sub-medians computed the same way:
  **−0.7800** on folds 11–25, **−0.6606** on folds 19–25 (E4).
- Fold boundaries revalidated index-identical to `make_sliding_folds` (as in Task 22).
- Mechanics mirror `env_bracket` formula-for-formula: entry `close + dir×half_cost`, exit
  `raw − dir×half_cost` (both legs at entry-bar ATR, half_cost = 0.03415×ATR), commission
  $0.01/trade, sizing `units = 0.005×equity/(1.0×ATR_entry)`, per-fold equity restart at 10k,
  MTM = realized + unrealized at each H1 close.
- Deterministic (ridge, no seeds): one run is the complete answer — no seed-median dimension
  (the anchor's is median-of-5 with seed-IQR 0.419; keep that in mind when comparing).
- Runtime 11 s; artifacts in `enhancements/09_probe/`.

## 2. Headline table (the anchor's exact metric: stitched return ÷ |MTM maxDD|)

| cell | folds | metric | return | maxDD MTM | PF | trade-Sharpe | gate | E4 metric / return |
|---|---|---|---|---|---|---|---|---|
| **anchor (RL, median-of-5)** | 1–25 | **−0.5260** | — | — | — | — | 0/5 | −0.6606 / — |
| V1·5y (time exit) | 1–25 | **−0.0159** | −0.8% | −51.6% | 1.009 | +0.06 | FAIL (14/25, floor 0.79) | −0.2571 / −8.6% |
| V2·5y (canonical bracket) | 1–25 | **−0.9676** | −85.3% | −88.2% | 0.895 | −1.30 | FAIL (3/25) | −0.8539 / −45.0% |
| **anchor (RL)** | 11–25 | **−0.7800** | — | — | — | — | — | −0.6606 |
| V1·5y | 11–25 | +0.0277 | +1.3% | — | 1.009 | +0.09 | — | −0.2571 / −8.6% |
| **V1·10y** | 11–25 | **+4.2626** | **+126.2%** | −29.6% | 1.113 | +0.81 | **FAIL (floor 0.82; breadth 11/15 OK, Sharpe OK)** | **+5.4268 / +80.3%** |
| V2·5y | 11–25 | −0.8875 | −61.2% | — | 0.911 | −1.08 | — | −0.8539 / −45.0% |
| V2·10y | 11–25 | −0.7247 | −43.1% | −59.5% | 0.952 | −0.65 | FAIL (5/15) | −0.4818 / −17.6% |

Per-era fold-metric medians (E4 reporting per the ratified rule): V1·5y 0.47 / −0.01 / 0.26 /
**−0.08** (E1–E4); V1·10y — / 1.86 / −0.12 / **+0.60**; V2 negative in every era both arms.

## 3. Finding 1 — the widening rescue converts to money

Same 15 folds, same entries rule, only the train window changes: **+1.3% → +126.2%**
(metric +0.03 → +4.26). Cost/gross drops 93.8% → 55.9% (the anchor ran ~111%). E4 flips
−8.6% → **+80.3%**. Not outlier-driven: 11/15 folds positive, 6/7 in E4 (f19 +15.0%, f25 +39.5%;
only f24 −11.4%); weakest era is E3 (metric median −0.12, f15/f16/f18 negative) — exactly the
era the Task-22 probe said degrades slightly at 10y. The supervised evidence chain
(IC → capture → ruler) is now consistent end-to-end.

## 4. Finding 2 — execution geometry: brackets destroy this signal

V2 shares V1's entries; only the exit differs. Result: 50/50 SL-vs-TP resolution (3404:3226 at
5y), win rate 48.7–50.1% vs the 53.4% the ±1×ATR bracket must clear, average resolution ~2 bars
— the bracket cashes out cost every ~2–4 bars before the 4-bar drift can play out. Cost/gross
532% (5y) / 157% (10y). The pessimistic both-touch→SL convention is NOT the cause: ambiguous
bars are 117/6702 = **1.7%** of V2 trades (53/3852 at 10y) — flipping every one of them to
TP-first could not rescue a 0.895 PF.

**Why this matters for the env:** the action space mandates a bracket on every position
(SL∈{1,1.5,2}×ATR, TP∈{1,1.5,2,3}R). A time-exit policy is only expressible as "pick the widest
bracket, learn to manual-close at bar 4" — reachable (45% of anchor exits were flip/manual) but
it must be *learned* against always-armed brackets that truncate exactly the tail the signal
monetizes. The one execution mode the signal provably supports is the one the action space makes
hardest. Any post-A/B-#4 failure analysis starts here.

## 5. Finding 3 — the honest failure: no cell passes the gate

V1·10y passes breadth (11/15 ≥ ceil(70%×15)) and mean trade-Sharpe (+0.75) but fails the PF floor
(10th-pct fold PF 0.82 < 0.90). Under the ratified discipline that is a FAIL — the gate is doing
its job (fat left tail: f16 −13.3%, f24 −11.4%). Also honest: V1 carries no stop (pinned pure
time exit) → −29.6% MTM maxDD even in the winning cell, −51.6% at 5y; the metric already prices
that, but a deployable variant would need a risk overlay, which is out of scope and unpinned.

Why V1·5y ≈ 0 when the Task-22 probe cleared at 5y in E1–E3: deployment mechanics. The probe's
capture is per-selected-bar with overlap; the deployable rule trades a correlated subset (~45%
of selected bars are skipped in-position: 7,890 of ~17.6k), compounds losses fold-to-fold, and
pays the full DD path. That conversion loss — ceiling ≈ +0.09 ATR/trade shrinking to PF 1.009 —
is precisely what this task existed to measure.

## 6. The benchmark A/B #4 must now beat

On folds 11–25 (A/B #4's grid), the standing numbers are:

| bar | metric | context |
|---|---|---|
| anchor RL (median-of-5) | −0.78 | what PPO does today on these folds |
| trivial bar (ship rule) | anchor + ratified Δ | necessary but no longer sufficient |
| **supervised V1·10y** | **+4.26 (+126%, PF 1.11, E4 +5.43)** | **a 26-line deterministic ridge rule with zero seeds and zero search** |

An RL candidate that ships under the ratified rule but lands far below +4.26 has not justified
its complexity over the supervised rule — the reviewer should weigh that explicitly at A/B #4
judgment time. (The supervised rule itself remains gate-FAIL on the PF floor, so it is a
*benchmark*, not the alternative deployment.)

Implication for A/B #4 design (unchanged, not launched): 10y training attacks the data side this
task just validated end-to-end; but V2·10y (−43%) shows more data does NOT fix bracket-mode
monetization — if PPO-with-brackets fails at 10y, the V1-vs-V2 gap is the pre-written post-mortem,
and the follow-on question becomes action-space/exit design, not more data.

## 7. Honesty box

- **Fills are H1-touch, not the env's M1 walk** (pinned): pessimism bounded by the 1.7%/1.4%
  ambiguous-bar rates (V2 only; V1 has no intrabar exits). Gap-throughs fill at the open (N1
  mirror): 52 gap exits at 5y, 49 at 10y.
- **No seed dimension** — ridge is deterministic; the anchor comparison point is a median over 5
  seeds with IQR 0.419. The +4.26 vs −0.78 gap (≈12×IQR) dwarfs that, but cell-level small
  differences would not.
- **Two variants × two arms, all pre-pinned, all reported** — including both catastrophic V2
  cells and the V1·5y near-zero. No cell was dropped or re-run.
- **15 folds only** in the 10y arm (structural: 10y history requires f11+); E1 is untestable
  at 10y.
- V1·10y's +126% concentrates in E2+E4; E3 is negative — regime dependence is real and the gate's
  PF floor correctly flags it.
- The lockbox (2024-07→) remains sealed; nothing here peeked past 2024-01-16.

## 8. Files

`enhancements/09_probe/`: `run_supervised_baseline.py` (pre-committed `bf4337a`),
`report.json` (all cells incl. anchor sub-stitches), `summary_*.csv` (per-fold rows),
`stitched_*.csv.gz` (equity curves), `trades_*.csv.gz`, `run_stdout.log`.
Protocol pin + results: EXECUTION_LOG Task 23.

*No training was run; the anchor stands; running-best is unchanged. Stopped for review —
A/B #4 launches only on reviewer go.*
