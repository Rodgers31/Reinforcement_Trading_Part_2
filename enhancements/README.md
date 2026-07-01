# Enhancements

Design and planning docs for improving the XAUUSD RL bracket-trading bot.
These are **planning artifacts**, not code. Nothing here changes the pipeline
until a change is explicitly implemented and reviewed.

Every claim in these docs was **verified against the source** (executed or traced
line-by-line), not taken from the README/docstrings — several of which are stale
(see `01-baseline-review-and-known-issues.md`).

## Index

| Doc | What it is |
|---|---|
| [`01-baseline-review-and-known-issues.md`](01-baseline-review-and-known-issues.md) | Verified explanation of what the bot does today + a prioritized list of bugs, doc-vs-code mismatches, and realism caveats found in the 2026-06 baseline review. |
| [`02-multi-instrument-training-plan.md`](02-multi-instrument-training-plan.md) | Extend the single-instrument (gold) bot to multiple FX pairs + metals. Answers "separate models vs. one mixed model," how it would learn, a phased plan, and the exact code changes. |
| [`03-rl-core-enhancements.md`](03-rl-core-enhancements.md) | Making the RL system *itself* smarter, more sample- and compute-efficient, and better at generalizing: a verified current-state anchor, a north-star vision, and a prioritized, phased assessment across 9 dimensions (representation, architecture, algorithm, reward, actions, compute, generalization, selection, realism) — with explicit "keep as-is" verdicts. |
| [`04-data-and-execution-plan.md`](04-data-and-execution-plan.md) | The operational sequence that ties 01–03 together: where to source data (hybrid Dukascopy backbone + OANDA recent/spreads/live, verified against the on-disk data), how far back / what format, then the ordered build-and-test ladder (build the measurement ruler → baseline → one change at a time → multi-instrument → reveal the holdout once). |

## Reading order

1. Start with **01** to understand the current system and the issues that shape
   every enhancement (especially the thin-signal overfitting risk, the
   single-instrument cost model, and the reporting-layer bugs B1–B3).
2. Then **02** for the multi-instrument direction (the trajectory / deployment shape).
3. Then **03** for improving the RL core itself. It treats multi-instrument (02) as
   a compatibility lens, not the subject, and depends on the doc-01 B1–B3 fixes as a
   prerequisite.
4. Then **04** for the execution plan — the concrete data-sourcing + step ordering
   that operationalizes 01–03. Read 01 → 02 → 03 → 04; **execute** in the order 04
   lays out (which defers to 03 §6 for the enhancement sequence).

## Status

- Author: baseline review + design pass, 2026-06.
- Nothing here is implemented yet. The end-to-end execution order lives in **doc 04
  §2** (build the ruler → baseline → one change at a time → multi-instrument → reveal
  once); it defers to doc 03 §6 for the enhancement sequence, doc 02 §6–§7 for
  multi-instrument phasing, and doc 03 §7 for the shared OOS validation harness. All
  treat the doc-01 B1–B3 reporting fixes as the first prerequisite (they're the
  "ruler," fixed before any baseline).
- Data: the backbone (~20y XAUUSD M1) comes from Dukascopy; OANDA (the user's puller)
  supplies the recent tail + real spreads + live parity. The on-disk OANDA data is
  ~3.5y (2023→2026) — enough for a smoke-test, not the full sliding walk-forward. See
  doc 04 §1.
- Environment note: this machine (Python 3.9) is missing `gymnasium`,
  `stable_baselines3`, `torch`, and `plotly`, so the RL pipeline can't run here;
  only the pandas/numpy core (data/features/metrics) executes. The locally-runnable
  first steps are deliberately pure-pandas: doc 02's Phase-0 similarity diagnostic,
  doc 03's MTM-vs-realized drawdown-gap measurement (§3.9a / §9), and doc 04's
  data-format/splice checks (§1).
