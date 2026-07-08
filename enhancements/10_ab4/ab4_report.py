"""Task 24 — A/B #4 ship-rule report (bespoke pairing; constants from ab_report).

Vanilla ab_report.compare would mispair a 15-fold candidate against the 25-fold
anchor (positional zip) and would compare against the anchor's FULL-25 median.
This report applies the SAME ratified legs with the Task-24 pinned mapping:

  candidate fold j  <->  anchor fold j+10   (same seed)
  anchor comparator = anchor run re-stitched over folds 11-25 per seed
                      (method proven in Task 23: full-25 re-stitch reproduced
                      the ratified per-seed metrics to <5e-4)
  E4 = candidate folds 9-15 (std f19-25) vs anchor folds 19-25, re-stitched.

Ship-rule constants are IMPORTED from ab_report.py (unchanged): leg a
delta-median >= 0.21, leg b paired Wilcoxon p < 0.01 with positive median,
leg c >= 4/5 seeds beat (finalist stage only), leg d no gate-leg regression.
The +4.26 supervised benchmark (Task 23) is recorded alongside as the
pre-committed JUDGMENT key — it is NOT a ship-rule leg.

Usage: .venv/bin/python enhancements/10_ab4/ab4_report.py --candidate runs/<ab4-run-dir>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scipy.stats import wilcoxon  # noqa: E402

from ab_report import (  # noqa: E402  — ratified constants + leg parser, unchanged
    SHIP_DELTA_MEDIAN_ABS, SHIP_WILCOXON_P, SHIP_MIN_SEEDS_BEAT,
    SHIP_N_SEEDS_FINAL, _parse_gate_legs,
)
from config import CFG  # noqa: E402
from eval_harness import metric_return_over_mtm_dd  # noqa: E402
from run_baseline import _stitch  # noqa: E402
from train_ppo import _passes_consistency_gate  # noqa: E402

ANCHOR_RUN = REPO / "runs" / "20260703-021710_b9bc9d6_baseline-3seed"
FOLD_OFFSET = 10                      # candidate fold j == standard fold j+10
STD_FOLDS = list(range(11, 26))       # anchor-side folds
E4_STD = list(range(19, 26))          # standard E4 folds
E4_CAND = list(range(9, 16))          # same folds in candidate numbering
SUPERVISED_BENCHMARK = {"V1_10y_metric": 4.2626, "V1_10y_E4_metric": 5.4268,
                        "source": "enh/09 Task 23 (folds 11-25)"}


def _anchor_side(seeds: list[int]) -> dict:
    """Anchor per-seed sub-metrics, fold metrics, gate legs on folds 11-25."""
    out = {}
    for s in seeds:
        summ = pd.read_csv(ANCHOR_RUN / f"seed_{s}_summary.csv")
        sub = summ[summ.fold.isin(STD_FOLDS)].reset_index(drop=True)
        eqs = [pd.read_csv(ANCHOR_RUN / "jobs" / f"f{k:02d}_s{s}" / "test_equity.csv",
                           index_col=0) for k in STD_FOLDS]
        e4 = [pd.read_csv(ANCHOR_RUN / "jobs" / f"f{k:02d}_s{s}" / "test_equity.csv",
                          index_col=0) for k in E4_STD]
        passed, detail = _passes_consistency_gate(
            sub, ret_col="test_return_pct", pf_col="test_profit_factor",
            sharpe_col="test_sharpe_trade")
        out[s] = {
            "metric_sub": metric_return_over_mtm_dd(_stitch(eqs), CFG.initial_equity),
            "metric_e4": metric_return_over_mtm_dd(_stitch(e4), CFG.initial_equity),
            "fold_metrics": dict(zip(sub.fold.tolist(), sub.fold_metric.tolist())),
            "gate_passed": bool(passed),
            "gate_detail": detail,
        }
    return out


def _candidate_e4(cand_dir: Path, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        eqs = [pd.read_csv(cand_dir / "jobs" / f"f{k:02d}_s{s}" / "test_equity.csv",
                           index_col=0) for k in E4_CAND]
        out[s] = metric_return_over_mtm_dd(_stitch(eqs), CFG.initial_equity)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    cand_dir = Path(args.candidate)
    cand = json.loads((cand_dir / "baseline_report.json").read_text())
    assert cand["n_folds"] == 15, f"expected 15-fold A/B #4 run, got {cand['n_folds']}"
    reg = json.loads((cand_dir / "registry.json").read_text())
    assert reg.get("sliding_train_years") == 10.0, "candidate run is not the 10y variant"

    seeds = [int(s) for s in cand["seeds"]]
    is_final = len(seeds) >= SHIP_N_SEEDS_FINAL
    anch = _anchor_side(seeds)
    cand_e4 = _candidate_e4(cand_dir, seeds)

    cand_median = statistics.median(cand["metric_per_seed"][str(s)] for s in seeds)
    anch_median = statistics.median(anch[s]["metric_sub"] for s in seeds)
    delta_median = cand_median - anch_median

    # paired per-(fold, seed) deltas: candidate fold j <-> anchor fold j+10
    deltas = []
    for s in seeds:
        cf = cand["fold_metrics_per_seed"][str(s)]          # 15 entries, folds 1..15
        af = anch[s]["fold_metrics"]                        # keyed by std fold 11..25
        deltas += [cf[j - 1] - af[j + FOLD_OFFSET] for j in range(1, 16)]
    median_delta = statistics.median(deltas)
    try:
        _, p_two = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        p_two = 1.0

    per_seed_beat = {s: cand["metric_per_seed"][str(s)] > anch_median for s in seeds}
    n_beat = sum(per_seed_beat.values())

    regressions = []
    for s in seeds:
        a_legs = _parse_gate_legs(anch[s]["gate_detail"])
        c_legs = _parse_gate_legs(cand["per_seed"][str(s)]["gate_detail"])
        for label, a_ok in a_legs.items():
            if a_ok and not c_legs.get(label, False):
                regressions.append({"seed": s, "leg": label})

    leg_a = delta_median >= SHIP_DELTA_MEDIAN_ABS
    leg_b = (median_delta > 0) and (p_two < SHIP_WILCOXON_P)
    leg_c = n_beat >= SHIP_MIN_SEEDS_BEAT
    leg_d = len(regressions) == 0
    ship = all([leg_a, leg_b, leg_c, leg_d])

    report = {
        "candidate": str(cand_dir),
        "stage": "5-seed FINALIST" if is_final else f"{len(seeds)}-seed PREVIEW",
        "pairing": f"candidate fold j <-> anchor fold j+{FOLD_OFFSET}, folds 11-25",
        "cand_median": round(cand_median, 4),
        "anchor_sub_median_same_seeds": round(anch_median, 4),
        "delta_median": round(delta_median, 4),
        "ship_rule_legs": {
            "a_delta_median>=0.21": {"pass": bool(leg_a), "delta_median": round(delta_median, 4)},
            "b_wilcoxon_p<0.01_pos": {"pass": bool(leg_b), "p_two_sided": round(float(p_two), 5),
                                      "median_paired_delta": round(median_delta, 4),
                                      "n_pairs": len(deltas)},
            "c_>=4of5_seeds_beat": {"pass": bool(leg_c), "n_beat": n_beat,
                                    "n_seeds": len(seeds),
                                    "note": "informational at preview stage",
                                    "per_seed_metric": {str(s): round(cand["metric_per_seed"][str(s)], 4)
                                                        for s in seeds}},
            "d_no_gate_regression": {"pass": bool(leg_d), "regressions": regressions},
        },
        "SHIP" if is_final else "SHIP_preview": bool(ship),
        "E4_subset_f19_25": {
            "candidate_per_seed": {str(s): round(cand_e4[s], 4) for s in seeds},
            "candidate_median": round(statistics.median(cand_e4.values()), 4),
            "anchor_per_seed": {str(s): round(anch[s]["metric_e4"], 4) for s in seeds},
            "anchor_median": round(statistics.median(a["metric_e4"] for a in anch.values()), 4),
        },
        "supervised_benchmark_judgment_key": SUPERVISED_BENCHMARK,
        "anchor_gate_passed": {str(s): anch[s]["gate_passed"] for s in seeds},
        "cand_gate_passed": {str(s): bool(cand["per_seed"][str(s)]["gate_passed"]) for s in seeds},
    }
    out_path = cand_dir / "ab4_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
