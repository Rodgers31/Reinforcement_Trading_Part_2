# Run index

Append-only. One line per full training run (see run_registry.py).

**Running best:** `20260703-021710_b9bc9d6_baseline-3seed` (baseline-3seed) — baseline anchor, metric median -0.5260 (ratified median-of-5)

| date (UTC) | git sha | label | metric median | gate | verdict |
|---|---|---|---|---|---|
| 2026-07-03 | 033e779-dirty | sizing-run | +2.7122 | PASS | sizing (1 fold, 1 seed — NOT a baseline) |
| 2026-07-03 | b9bc9d6 | baseline-3seed | -0.5282 | FAIL | interim 3-seed pass — SUPERSEDED by the 5-seed row below |
| 2026-07-03 | b9bc9d6 | baseline-3seed | -0.5260 | FAIL | **BASELINE ANCHOR** (ratified median-of-5); gates 0/5; seed-IQR 0.419 |
| 2026-07-04 | 53fe6de | turnover-p045-3seed | +1.2810 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.045 (3-seed PREVIEW) — SUPERSEDED by the 5-seed finalist row below |
| 2026-07-04 | 53fe6de | turnover-p045-3seed | +1.2810 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.045 — SAME run EXTENDED to 5 seeds (ratified median-of-5); gates 0/5; **NO SHIP** (ship rule 3/4; Wilcoxon p=0.020 fails <0.01); running-best UNCHANGED |
| 2026-07-04 | 49a9ef9 | turnover-p022-3seed | -0.1687 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.022 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
| 2026-07-05 | c2eb933 | costrand-p40 | -0.5503 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.0 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
| 2026-07-05 | 753beea | turnover-flipaware-p045 | -0.4989 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.045 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
| 2026-07-07 | 4fa6046 | ab4-10y-3seed | -0.6078 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.0 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
| 2026-07-07 | 1142ac3 | ab5-holdhorizon-10y-3seed | -0.6060 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.0 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
| 2026-07-07 | d7b5ed4 | ab5b-holdhorizon-10y-6M-3seed | -0.5011 | FAIL | Phase-C CANDIDATE turnover_penalty_r=0.0 (PROVISIONAL median-of-3); gates 0/3; running-best UNCHANGED (ship decided by ab_report.py) |
