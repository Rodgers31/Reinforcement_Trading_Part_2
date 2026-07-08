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

---

## RESULTS (executed 2026-07-08; pre-flight reproduced committed V1·10y exactly)

| cell | metric | return (viab ≥+63.1%) | PF | gate legs (breadth/floor/Sharpe) | E4 ret | beats-beta | FULL |
|---|---|---|---|---|---|---|---|
| BASE (control) | +4.2626 | +126.2% ✓ | 1.113 | 11/15 ✓ / **0.82 ✗** / +0.75 ✓ | +80.3% | ✓ | ✗ |
| L1 shorts-q90 | +4.0381 | +91.1% ✓ | 1.117 | **10/15 ✗** / 0.84 ✗ / +0.66 ✓ | +51.5% | ✓ | ✗ |
| **L2 long-only** | **+6.2523** | +89.7% ✓ | **1.157** | 11/15 ✓ / **0.88 ✗** / +0.80 ✓ | +50.5% | ✓ | **✗** |
| BETA always-long | −0.9826 | −92.8% ✗ | 0.912 | 2/15 / 0.80 / −1.17 | −79.3% | — | — |

Context row: unlevered gold B&H = +53.3% over the stitched window, +4.8% over the E4 window.
BETA (long exposure through the trade grammar) is annihilated by cost drag (10,310 trades) —
the cells' returns are selection skill, not beta.

**The diagnosed mechanism worked — and the floor still held.** L2 flipped both diagnosed
short-bleed victims positive (f15 −4.8%→+1.2% PF 1.047; f24 −11.4%→+2.4% PF 1.086), raised the
floor 0.82→0.88, PF to 1.157, and posted the best metric ever measured on this dev surface
(+6.25). The remaining floor drivers are f16 (0.850) and f18 (0.842) — E3 chop folds where the
diagnosis says the signal itself weakens (enh/11 Q3): not a side problem, not a tail problem,
not a leverage problem. L1's conviction-gated shorts kept most of the bleed and lost a breadth
fold (10/15) — strictly dominated by L2.

**TERMINAL RULE FIRES (pre-committed §3): neither L1 nor L2 passes → the XAUUSD-alone supervised
line RESTS at benchmark status.** No L3, no constant tuning, no threshold search. L2's 0.88 vs
0.90 miss is two folds of genuinely weak-signal regime; making the floor by construction would
require exactly the kind of post-hoc lever this document forbids. The benchmark family is now:
V1·10y +4.26/+126.2% (primary, as pinned in Task 23) with L2 long-only +6.25/+89.7% recorded
alongside as the best-measured cell of the closed 14-cell ledger.

Artifacts: `enhancements/12_levers/` (run_levers.py, levers_report.json, per-cell CSVs).
