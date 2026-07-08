# 12 — Final XAUUSD Supervised Lever Family: Asymmetric / Long-Only (Track A)

**Date:** 2026-07-08 · **Branch:** `execution/phase-a` · **Status: PROPOSAL pinned before
execution (Task 27); DESIGN THEN EXECUTE per reviewer direction.**
**Base:** Task-23 **V1·10y** (fwd4·ridge, train-q80 |score| top-quintile, sign direction, 4-bar
time exit, 10y train, folds 11–25): +4.2626 / +126.2% / PF 1.113 / E4 +80.3%; fails only the
PF-floor gate leg (0.82 < 0.90). **Motivating diagnosis (enh/11 §RESULTS, already on record):**
E3 weakness is a SHORT-side bleed (shorts net-negative in 4 of 6 E3 folds) in low-vol chop;
risk overlays (H1–H3) could not fix the floor. This family attacks the diagnosed mechanism
directly — the last XAUUSD-alone supervised lever family.

## 1. Pre-declared cells (exactly two; zero new constants)

- **L1 — asymmetric entry threshold:** longs unchanged (enter when score>0 and |score| ≥
  train-q80); **shorts require |score| ≥ train-q90** (the probe's existing top-decile cut,
  Task-22 pin — an existing constant, not a new one). Mechanism: demand more conviction from the
  diagnosed-weak side without killing it.
- **L2 — long-only:** longs unchanged; shorts never taken. Zero constants of any kind.

Everything else frozen: fwd4·ridge fit, 10y windows, k=4 time exit, sizing 0.5% at 1×ATR risk
unit, 0.0683×ATR cost, folds 11–25, ruler + gate functions. No stop/sizing overlays (that line
is closed, enh/11).

## 2. Mandatory beta control (reported in every table)

**BETA = always-long in the identical trade grammar:** enter long at EVERY bar when flat, k=4
time exit, same sizing/costs/folds/ruler — long gold exposure expressed through the exact
machinery the cells use, so "beats beta" compares selection skill, not mechanics. Secondary
context row (informational, not a verdict input): unlevered gold buy-&-hold over the stitched
window. **Beats-beta leg = cell stitched return > BETA's AND cell metric > BETA's** (both, so a
cell cannot pass on a DD artifact).

## 3. Success criterion + TERMINAL RULE (pre-committed)

A cell passes only if ALL of: **(a)** full ratified gate on folds 11–25 (breadth ≥11/15, 10th-pct
fold PF ≥ 0.90, mean trade-Sharpe > 0); **(b)** viability floor — stitched return ≥ **+63.1%**;
**(c)** E4-subset return > 0; **(d)** beats-beta (both legs of §2). Ranking among passers:
highest 10th-pct fold PF (anti-self-flattery, as enh/11 §3).
**TERMINAL RULE: if neither L1 nor L2 passes, the XAUUSD-alone supervised line RESTS at
benchmark status — no L3, no constant tuning, no threshold search.** Any XAUUSD revival after
that requires new information from OUTSIDE this dev surface (e.g. Track B replication evidence),
under a fresh pin.

## 4. Multiplicity ledger update

Supervised ruler-cells to date: 12 (Task 23: 6; Task 24 V3: 3; enh/11: 3). **This proposal adds
exactly 2 → 14.** (BETA is a control, not a searched cell.) Breadth-leg luck at n=15 ≈ 5.9% per
cell; the four-condition pass bar (gate ∧ viability ∧ E4 ∧ beats-beta) is far stricter, and the
terminal rule stops the ledger here for XAUUSD-alone.

## 5. Execution

Runner `enhancements/12_levers/run_levers.py`: per-side selection arrays; **pre-flight must
reproduce the committed V1·10y per-fold rows exactly** (q80 both sides) before any cell runs;
then BASE / L1 / L2 / BETA, each with the full ruler row, all gate legs, E4 subset, per-era
medians, and the f15/f16/f18/f24 fold table. Results appended below; artifacts in
`enhancements/12_levers/`. Deterministic, no seeds.

*Pinned before execution; see EXECUTION_LOG Task 27 (which also pins Tracks B and C and the
project endgame).*
