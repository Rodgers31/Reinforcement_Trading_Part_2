# Run index

Append-only. One line per full training run (see run_registry.py).

**Running best:** `20260703-021710_b9bc9d6_baseline-3seed` (baseline-3seed) — baseline anchor, metric median -0.5260 (ratified median-of-5)

| date (UTC) | git sha | label | metric median | gate | verdict |
|---|---|---|---|---|---|
| 2026-07-03 | 033e779-dirty | sizing-run | +2.7122 | PASS | sizing (1 fold, 1 seed — NOT a baseline) |
| 2026-07-03 | b9bc9d6 | baseline-3seed | -0.5282 | FAIL | interim 3-seed pass — SUPERSEDED by the 5-seed row below |
| 2026-07-03 | b9bc9d6 | baseline-3seed | -0.5260 | FAIL | **BASELINE ANCHOR** (ratified median-of-5); gates 0/5; seed-IQR 0.419 |
