"""Phase-C A/B comparator + RATIFIED ship-rule evaluator.

Compares a CANDIDATE run (a `run_baseline.py` output produced with ONE changed
knob) against the ANCHOR / running-best. READ-ONLY: it never touches the system
under test — it only reads each run's `baseline_report.json`, `seed_*_summary.csv`,
and per-job `test_trades.csv` / `job.json`.

RATIFIED SHIP RULE  (2026-07-03 — final; spends the reserved one-time
recalibration of Phase-B's noise-sized +10% relative margin). Ship the candidate
over the running-best iff ALL FOUR legs pass:

  (a) Δmedian  >= 0.21  ABSOLUTE over running best
        (= 0.5 × baseline seed-IQR 0.4188, pinned here as a NUMBER, not recomputed)
  (b) paired per-(fold,seed) Wilcoxon signed-rank  p < 0.01  AND positive median delta
  (c) >= 4 / 5 seeds beat the running-best median
  (d) no gate regression  (no gate leg flips OK -> FAIL vs the anchor, per seed)

The 3-seed A/B yields a PREVIEW (legs evaluated on the ranked seeds present); the
5-seed finalist evaluates the full rule. Turnover diagnostics prove whether the
mechanism engaged: did turnover drop AND did NET improve?

Usage:
  .venv/bin/python ab_report.py --candidate runs/<cand_dir> [--anchor runs/<dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR = "runs/20260703-021710_b9bc9d6_baseline-3seed"

# ── RATIFIED ship-rule constants (pinned NUMBERS — do not recompute per run) ──
SHIP_DELTA_MEDIAN_ABS = 0.21      # 0.5 × baseline seed-IQR 0.4188
SHIP_WILCOXON_P       = 0.01
SHIP_MIN_SEEDS_BEAT   = 4          # of the 5-seed finalist
SHIP_N_SEEDS_FINAL    = 5


def _load_report(run_dir: Path) -> dict:
    return json.loads((run_dir / "baseline_report.json").read_text())


def _parse_gate_legs(gate_detail: list[str]) -> dict[str, bool]:
    """Map each gate leg's label -> passed?, parsing the '[OK  ]'/'[FAIL]' prefix."""
    legs = {}
    for line in gate_detail:
        passed = line.strip().startswith("[OK")
        label = line.split("]", 1)[1].split(":")[0].strip() if "]" in line else line
        legs[label] = passed
    return legs


def _turnover_diag(run_dir: Path, seeds: list[int], n_folds: int) -> dict:
    """Aggregate turnover / cost diagnostics over the given seeds' TEST trades.

    cost_cash per trade = (spread_atr_frac + 2·slippage_atr_frac)·entry_atr·units
    (the round-trip cost baked into fill prices by env_bracket._half_cost);
    gross = net + cost. Mirrors the RF-1 'cost as % of gross' definition.
    """
    import pandas as pd
    from config import CFG

    rt_cost_frac = CFG.spread_atr_frac + 2.0 * CFG.slippage_atr_frac
    net = cost = 0.0
    n_trades = n_flip = 0
    bdays = 0
    per_fold_counts = []
    for s in seeds:
        for k in range(1, n_folds + 1):
            jd = run_dir / "jobs" / f"f{k:02d}_s{s}"
            tf = jd / "test_trades.csv"
            if not tf.exists() or tf.stat().st_size <= 5:
                per_fold_counts.append(0)
                continue
            d = pd.read_csv(tf)
            per_fold_counts.append(len(d))
            if not len(d):
                continue
            net += float(d["pnl"].sum())
            cost += float((rt_cost_frac * d["entry_atr"].astype(float)
                           * d["units"].astype(float)).sum())
            n_trades += len(d)
            if "exit_reason" in d:
                n_flip += int((d["exit_reason"] == "flip_close").sum())
            job = json.loads((jd / "job.json").read_text())
            start, end = job["test"].split("->")
            bdays += len(pd.bdate_range(start, end))
    gross = net + cost
    return {
        "n_trades": n_trades,
        "trades_per_day": round(n_trades / max(bdays, 1), 4),
        "mean_trades_per_fold": round(n_trades / max(len(per_fold_counts), 1), 1),
        "flip_pct": round(100.0 * n_flip / max(n_trades, 1), 2),
        "net_pnl": round(net, 2),
        "gross_pnl": round(gross, 2),
        "cost_cash": round(cost, 2),
        "cost_pct_of_gross": (round(100.0 * cost / gross, 1) if gross > 0 else None),
    }


def compare(candidate_dir: Path, anchor_dir: Path) -> dict:
    from scipy.stats import wilcoxon

    cand = _load_report(candidate_dir)
    anch = _load_report(anchor_dir)

    anchor_median = anch["metric_median"]              # running-best median
    cand_median   = cand["metric_median"]
    delta_median  = cand_median - anchor_median

    cand_seeds = [int(s) for s in cand["seeds"]]
    anch_seeds = [int(s) for s in anch["seeds"]]
    shared = sorted(set(cand_seeds) & set(anch_seeds))
    is_final = len(cand_seeds) >= SHIP_N_SEEDS_FINAL

    # ── paired per-(fold,seed) deltas on shared seeds ──
    deltas = []
    for s in shared:
        cf = cand["fold_metrics_per_seed"][str(s)]
        af = anch["fold_metrics_per_seed"][str(s)]
        deltas += [c - a for c, a in zip(cf, af)]
    import statistics
    median_delta = statistics.median(deltas) if deltas else 0.0
    if deltas and any(abs(d) > 1e-12 for d in deltas):
        try:
            _, p_two = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p_two = 1.0
    else:
        p_two = 1.0  # all-zero (e.g. penalty=0 invariant): no difference

    # ── per-seed: candidate metric vs running-best median ──
    per_seed_beat = {s: cand["metric_per_seed"][str(s)] > anchor_median for s in cand_seeds}
    n_beat = sum(per_seed_beat.values())

    # ── gate regression: no leg OK->FAIL vs anchor, per shared seed ──
    regressions = []
    for s in shared:
        a_legs = _parse_gate_legs(anch["per_seed"][str(s)]["gate_detail"])
        c_legs = _parse_gate_legs(cand["per_seed"][str(s)]["gate_detail"])
        for label, a_ok in a_legs.items():
            if a_ok and not c_legs.get(label, False):
                regressions.append({"seed": s, "leg": label})
    cand_gate_pass = sum(cand["per_seed"][str(s)]["gate_passed"] for s in cand_seeds)
    anch_gate_pass = sum(anch["per_seed"][str(s)]["gate_passed"] for s in anch_seeds)

    # ── the four legs (preview if < 5 seeds) ──
    leg_a = delta_median >= SHIP_DELTA_MEDIAN_ABS
    leg_b = (median_delta > 0) and (p_two < SHIP_WILCOXON_P)
    leg_c = n_beat >= SHIP_MIN_SEEDS_BEAT
    leg_d = len(regressions) == 0
    legs = {
        "a_delta_median>=0.21": {"pass": bool(leg_a),
                                 "delta_median": round(delta_median, 4),
                                 "threshold": SHIP_DELTA_MEDIAN_ABS,
                                 "cand_median": round(cand_median, 4),
                                 "anchor_median": round(anchor_median, 4)},
        "b_wilcoxon_p<0.01_pos": {"pass": bool(leg_b),
                                  "p_two_sided": round(float(p_two), 5),
                                  "median_paired_delta": round(median_delta, 4),
                                  "n_pairs": len(deltas)},
        "c_>=4of5_seeds_beat_running_best": {"pass": bool(leg_c),
                                             "n_beat": n_beat,
                                             "n_seeds": len(cand_seeds),
                                             "need": SHIP_MIN_SEEDS_BEAT,
                                             "per_seed_metric": {str(s): round(cand["metric_per_seed"][str(s)], 4) for s in cand_seeds}},
        "d_no_gate_regression": {"pass": bool(leg_d),
                                 "regressions": regressions,
                                 "cand_gate_pass": cand_gate_pass,
                                 "anchor_gate_pass": anch_gate_pass},
    }
    ship = all([leg_a, leg_b, leg_c, leg_d])

    # ── turnover diagnostics: candidate vs anchor on the SHARED seeds ──
    n_folds = cand["n_folds"]
    cand_turn = _turnover_diag(candidate_dir, shared, n_folds)
    anch_turn = _turnover_diag(anchor_dir, shared, n_folds)

    return {
        "candidate": str(candidate_dir),
        "anchor": str(anchor_dir),
        "is_finalist": is_final,
        "stage": "5-seed FINALIST" if is_final else f"{len(cand_seeds)}-seed PREVIEW",
        "shared_seeds": shared,
        "cand_seeds": cand_seeds,
        "anchor_median_running_best": round(anchor_median, 4),
        "cand_median": round(cand_median, 4),
        "delta_median": round(delta_median, 4),
        "ship_rule_legs": legs,
        "SHIP" if is_final else "SHIP_preview": ship,
        "turnover": {
            "candidate": cand_turn,
            "anchor_same_seeds": anch_turn,
            "trades_per_day_delta": round(cand_turn["trades_per_day"] - anch_turn["trades_per_day"], 4),
            "flip_pct_delta": round(cand_turn["flip_pct"] - anch_turn["flip_pct"], 2),
            "cost_pct_of_gross_delta": (
                round(cand_turn["cost_pct_of_gross"] - anch_turn["cost_pct_of_gross"], 1)
                if (cand_turn["cost_pct_of_gross"] is not None and anch_turn["cost_pct_of_gross"] is not None)
                else None),
            "net_pnl_delta": round(cand_turn["net_pnl"] - anch_turn["net_pnl"], 2),
        },
    }


def _dir(delta: float, down_word: str, up_word: str) -> str:
    if abs(delta) < 1e-9:
        return "unchanged"
    return down_word if delta < 0 else up_word


def _fmt(report: dict) -> str:
    L = report["ship_rule_legs"]
    t = report["turnover"]
    ct, at = t["candidate"], t["anchor_same_seeds"]
    ship_key = "SHIP" if report["is_finalist"] else "SHIP_preview"
    lines = [
        f"===== Phase-C A/B report — {report['stage']} =====",
        f"candidate : {report['candidate']}",
        f"anchor    : {report['anchor']}  (running-best median {report['anchor_median_running_best']:+.4f})",
        f"shared seeds: {report['shared_seeds']}",
        "",
        f"median metric  candidate {report['cand_median']:+.4f}  vs anchor {report['anchor_median_running_best']:+.4f}"
        f"   Δ {report['delta_median']:+.4f}",
        "",
        "RATIFIED ship rule (all four legs must pass):",
        f"  (a) Δmedian >= {SHIP_DELTA_MEDIAN_ABS}          : {'PASS' if L['a_delta_median>=0.21']['pass'] else 'fail'}"
        f"   (Δ={L['a_delta_median>=0.21']['delta_median']:+.4f})",
        f"  (b) Wilcoxon p<{SHIP_WILCOXON_P} & median Δ>0 : {'PASS' if L['b_wilcoxon_p<0.01_pos']['pass'] else 'fail'}"
        f"   (p={L['b_wilcoxon_p<0.01_pos']['p_two_sided']}, medianΔ={L['b_wilcoxon_p<0.01_pos']['median_paired_delta']:+.4f}, n={L['b_wilcoxon_p<0.01_pos']['n_pairs']})",
        f"  (c) >= {SHIP_MIN_SEEDS_BEAT}/5 seeds beat median  : {'PASS' if L['c_>=4of5_seeds_beat_running_best']['pass'] else 'fail'}"
        f"   ({L['c_>=4of5_seeds_beat_running_best']['n_beat']}/{L['c_>=4of5_seeds_beat_running_best']['n_seeds']} beat {report['anchor_median_running_best']:+.4f})",
        f"  (d) no gate regression          : {'PASS' if L['d_no_gate_regression']['pass'] else 'fail'}"
        f"   (cand gates {L['d_no_gate_regression']['cand_gate_pass']}, anchor {L['d_no_gate_regression']['anchor_gate_pass']})",
        f"  => {ship_key}: {report[ship_key]}"
        + ("" if report["is_finalist"] else "   (preview only; finalist needs 5 seeds)"),
        "",
        "Turnover mechanism (candidate vs anchor, same seeds):",
        f"  trades/day      : {ct['trades_per_day']:.3f}  vs {at['trades_per_day']:.3f}   Δ {t['trades_per_day_delta']:+.3f}",
        f"  flip %          : {ct['flip_pct']:.2f}%  vs {at['flip_pct']:.2f}%   Δ {t['flip_pct_delta']:+.2f}",
        f"  cost % of gross : {ct['cost_pct_of_gross']}  vs {at['cost_pct_of_gross']}   Δ {t['cost_pct_of_gross_delta']}",
        f"  net PnL (cash)  : {ct['net_pnl']:+.2f}  vs {at['net_pnl']:+.2f}   Δ {t['net_pnl_delta']:+.2f}",
        f"  MECHANISM: turnover {_dir(t['trades_per_day_delta'], 'DROPPED', 'rose')}; "
        f"net {_dir(t['net_pnl_delta'], 'IMPROVED', 'worsened')}.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--out", default=None, help="path for ab_report.json (default: candidate dir)")
    args = ap.parse_args()

    cand_dir = Path(args.candidate)
    anch_dir = Path(args.anchor)
    report = compare(cand_dir, anch_dir)
    out = Path(args.out) if args.out else (cand_dir / "ab_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(_fmt(report))
    print(f"\n[ab_report] written to {out}")


if __name__ == "__main__":
    main()
