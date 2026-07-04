# 06 — Baseline trust investigation: verification scripts

Read-only analysis scripts that produced/verified the numbers in
[`../06-baseline-trust-investigation.md`](../06-baseline-trust-investigation.md).
None trains or evaluates a model; none touches sealed data (≥ 2024-07-01). Run
from the repo root with the project venv:

- **`passive_ruler_check.py`** — Agent C's ruler-fairness validator. Builds the
  SAME 25 sliding folds and the SAME cost env as production, then runs rule-based
  (NOT trained) passive policies (`ema_atr_trend`, long-only, random) on each
  fold's TEST window. Asserts the lockbox, writes `passive_*_perfold.csv` +
  `passive_summary.json` next to itself. Verdict: the ruler is fair (passive
  long-only wins 5/5 strong-bull folds, loses 3/3 bear folds; corr 0.896 w/ gold
  B&H). Reusable for future ruler audits.
- **`verify_claims.py`** — reproduces the 2023 measured ask−bid spread / ATR
  fraction (0.089–0.090 > charged 0.0623) and per-fold gold buy-and-hold facts.
- **`verify_A.py`** — reproduces the old-window (2024-02→2026-05) gold
  buy-and-hold (+124.05%) and the old absolute-cost constant.

  `.venv/bin/python enhancements/06_investigation/passive_ruler_check.py`
