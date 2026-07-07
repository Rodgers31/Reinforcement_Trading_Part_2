# 11 — Supervised Hardening Proposal

**Date:** 2026-07-07 · **Branch:** `execution/phase-a` · **Status: APPROVED 2026-07-07 with
reviewer amendments A1–A3 (incorporated below); execution pinned in EXECUTION_LOG Task 26.**
Originally written design-only while A/B #5 trained (Task 25), reviewed jointly with its result.
**Base system:** Task-23 **V1·10y** — fwd4·ridge top-quintile |score| (train-q80 threshold),
sign direction, 4-bar time exit, 10y train window, folds 11–25: metric **+4.2626**, +126.2%,
PF 1.113, trade-Sharpe +0.81, E4 +5.43/+80.3% — **gate FAIL on exactly one leg** (10th-pct fold
PF 0.82 < 0.90), and no protective stop (MTM maxDD −29.6%).

## 1. The gap to deployable

| ratified gate leg (folds 11–25) | V1·10y today | binding? |
|---|---|---|
| breadth: ≥ ceil(70%×15)=11 folds return>0 & PF>1 | **11/15 PASS** | no |
| floor: 10th-pct fold PF ≥ 0.90 | **0.82 FAIL** (f24 0.80, f16 0.81, f18 0.84) | **YES** |
| mean trade-Sharpe > 0 | **+0.75 PASS** | no |

The hardening question is narrow: **can a pre-declared risk overlay lift the worst-decile fold
PF above 0.90 without destroying what works** (the Task-23/24 evidence says tight brackets
destroy this signal, so any stop must be wide), while also giving the strategy a defined
per-trade risk bound (the current pure time exit has none — undeployable as-is regardless of
metric).

## 2. Pre-declared overlay cells (exact constants; declared BEFORE any cell runs)

All cells: base V1·10y unchanged (entries, threshold rule, k=4 time exit, folds 11–25, 0.0683
cost model, anchor ruler + gate). **Three cells, no variants of variants, no constant tuning.**

- **H1 — protective stop only:** SL at **2.0×ATR_entry** (the env menu's widest; the V3
  diagnostic measured that a 2×ATR stop coexists with this signal), intrabar H1-touch scan with
  the Task-23 fill conventions (gap-through at open, ambiguity pessimism), **no TP**, time exit
  at k=4 unchanged. Sizing unchanged (0.5% at 1×ATR risk unit — the stop changes realized risk,
  not the sizing rule, preserving comparability with the base).
  *Mechanism:* truncates the worst 4-bar excursions → targets the PF floor and the −29.6% MTM DD.
  *Known failure mode:* whipsaw conversion of eventual winners (V3 saw ~50% SL exits at 2×ATR
  over 24 bars; over only 4 bars the touch rate will be far lower).
- **H2 — vol-scaled sizing only:** per-entry risk budget `0.5% × min(1, m/atr_close[t])` where
  `m` = median of `atr_close` over that fold's OWN train window (train-derived, refit per fold,
  never from test), **floored at 0.25×** (never below quarter-size, never above 1×). No stop.
  *Mechanism:* de-levers high-vol regimes where the fat left-tail folds live; targets DD and
  fold-PF dispersion without touching trade selection.
- **H3 — both:** H1 stop + H2 sizing, constants identical to above.

Reported for every cell (and the unmodified base as control): the full ruler row (metric,
return, MTM maxDD, PF, trade-Sharpe, all three gate legs), per-era fold-metric medians,
**E4 subset** (ratified rule), exit-reason mix, ambiguity counts, and per-fold table for
f15/f16/f18/f24 specifically.

## 3. Success criterion + stop rule (pre-committed; amendments A1 + A3 ratified 2026-07-07)

A cell "passes" only if it **passes the FULL ratified gate on folds 11–25** (all three legs) AND
keeps the E4-subset return positive AND **(A1, economic-viability floor) keeps stitched return ≥
+63.1% — half the unhardened base's +126.2%**. A1 is deliberately a RETURN floor, not a metric
floor: a stop mechanically shrinks the metric's MTM-DD denominator, so a metric target would be
self-flattering; a hardening that buys its gate pass by giving up more than half the economics
has failed economically. Ranking among passers (if several): highest 10th-pct fold PF, not
highest metric (same anti-self-flattery reasoning; the metric is reported, never targeted).
**If none of H1/H2/H3 passes, the hardening line STOPS** — no H4, no constant adjustment, no
threshold revisit. The honest conclusion would then be: the supervised edge is real but not
deployable under the ratified gate, and the E3 diagnosis (§4) decides whether a *new*, separately
pinned proposal is even warranted.

**(A3) A passing cell does NOT proceed toward deployment directly.** It triggers a separately
pinned **enh/12 pre-deployment validation proposal** — a report-only robustness battery:
selection-threshold perturbation (q75/q85 around the frozen q80), ridge-alpha perturbation, and
a cost +25% stress. **The lockbox stays sealed until enh/12 exists and passes review.**

## 4. E3 failure-diagnosis plan (analysis-only; no parameter may change because of it)

**(A2, ratified 2026-07-07): this diagnosis runs FIRST — before any H-cell — so the cell results
are read as mechanism-confirmed or mechanism-suspicious against it.** The no-parameter-change
rule is unchanged: nothing in H1/H2/H3 may be altered in response to the diagnosis.

E3 (tests 2017-07→2020-07) is the base's weak era (fold-metric median −0.12; f15 −4.8%,
f16 −13.3%, f18 −8.7%). Pre-registered questions, all answerable from data already on disk:

- **Q1 — concentration:** share of each bad fold's loss carried by its 10 worst trades
  (trade-log decomposition). Tail-driven → H1 should mechanically fix it (verified by the H1
  cell); broad decay → regime problem, stops won't save it.
- **Q2 — directional asymmetry:** long-vs-short capture per E3 fold (2018-2020 = chop + USD
  strength; a short-side bleed would show here).
- **Q3 — signal presence:** univariate era-IC from the Task-22 `results_univariate.csv` (already
  computed): do the load-bearing features (dow_cos, tod_cos, lower_wick_ratio) hold their IC in
  E3? If yes → execution/regime issue; if no → E3 is a genuine no-signal era at this window,
  the analog of E4-at-5y (and the 15y-window question becomes a legitimate FUTURE pin, not part
  of this proposal).
- **Q4 — vol regime:** E3 bad-fold ATR%-of-price percentile vs the clearing folds (does H2's
  de-levering even engage there?).

Deliverable: a diagnosis section in the eventual results doc. Any new lever this suggests (e.g.
a regime sit-out filter) requires its **own** pre-registered proposal with fresh multiplicity
accounting — it may NOT be bolted onto this one.

## 5. Refit cadence (decided by argument, not search)

**Pinned: refit every 6 months — the fold grid's own step.** The +4.26 benchmark IS the
6-month-refit number (each fold refits on its 10y window); adopting any other cadence would
detach the live procedure from the measured evidence. Quarterly refit would require a new fold
grid (changes the ruler — rejected); annual refit saves nothing (a ridge refit costs seconds)
and doubles model staleness — rejected. Live procedure per refit: rebuild features on the
trailing 10y, fit `Ridge(alpha=1.0)` on train-standardized features, recompute the q80
|score| threshold on that window's own predictions — all frozen mechanics, zero discretion.

## 6. Anti-overfit discipline + multiplicity ledger

- **Frozen from prior pins:** 25 features, fwd4 label, ridge α=1.0, top-quintile q80 rule
  (train-derived), k=4, 10y window, folds 11–25, cost 0.0683×ATR, ruler + gate functions.
  **No threshold search anywhere in this proposal.** The three overlay constants (2.0×ATR,
  min(1, m/atr), 0.25 floor) are declared here, before any run.
- **Ledger:** supervised ruler-cells evaluated to date = 9 (Task 23: V1/V2 × 5y/10y + two
  sub-cuts; Task 24: V3 × 3). This proposal adds exactly **3**. Breadth-leg luck under a no-edge
  null at n=15 is P(≥11/15 | p=½) ≈ 5.9% per cell (all three gate legs jointly much lower);
  3 pre-declared cells keep the family-wise false-deploy risk ≲ a few percent — acceptable, and
  the stop rule (§3) prevents the ledger from growing silently.
- **E4 + per-era reporting mandatory** (ratified 2026-07-07); anchor sub-comparators −0.78
  (f11–25) / −0.66 (E4) and the V1·10y control reported in every table.
- **Lockbox scope guard:** no bar at/after 2024-07-01 is read anywhere in this proposal.
  (Sealing/reveal governance lives in the §3 A3 clause and the future enh/12 — not here.)
- Deterministic (no seeds); Task-23 fidelity caveats carry over (H1-touch fills bounded by
  measured ambiguity rates).

## 7. Execution plan and cost (approved 2026-07-07; runs in parallel with the A/B #5 6M pool)

Order per A2: **§4 E3 diagnosis first** (pure analysis over committed artifacts:
`09_probe/trades_V1_10y.csv.gz`, `summary_V1_10y.csv`, `08_probe/results_univariate.csv`,
`08_probe/fold_windows.csv`), **then** the three cells. Runner = new script under
`enhancements/11_hardening/` with a pre-flight that reproduces the committed V1·10y per-fold rows
exactly (stop disabled, scale=1) before any H-cell runs — proving the runner faithful to the
Task-23 simulator. Protocol pin in EXECUTION_LOG (Task 26) precedes execution per house rule.
One deterministic run: 3 cells × 15 folds, minutes of compute. Results = RESULTS section
appended to this file (no new doc number).

## 8. Relationship to the RL tracks

Independent of A/B #5's outcome by construction (different lever). If A/B #5 fails and the
Task-25 TERMINAL RULE fires (end-to-end RL closed as an architecture negative), this proposal
becomes the primary line toward a deployable system; the hybrid (supervised-signal-in-obs RL)
remains a separate future research item requiring its own proposal. If A/B #5 ships, this
proposal still stands — a deployable supervised book and an RL book are portfolio candidates,
not rivals (doc 02's portfolio framing).

*Nothing in this document has been executed. Awaiting joint review with the A/B #5 result.*
