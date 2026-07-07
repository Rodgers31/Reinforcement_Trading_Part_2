"""Phase-B baseline launcher: resumable fold×seed job pool + aggregation.

ANCHOR DISCIPLINE (binding, ratified 2026-07-03): this launcher only READS the
system under test (config/env/training code untouched). No config,
hyperparameter, cost, eligibility, or gate changes in response to anything the
run reveals — improvement ideas go to the Phase-C candidates list in
EXECUTION_LOG.md, nothing else. The completed baseline INITIALIZES the
running-best pointer by definition; gate_passed records deployability
separately (a gate-FAIL baseline is still the anchor).

Seed protocol (pre-declared BEFORE launch): ratified 5-seed set is
{42, 43, 44, 45, 46}; the provisional pass runs {42, 43, 44}; the extension
adds {45, 46} to the SAME parent run. The extension is never seed-picked.

Layout:  runs/<stamp>_<sha>_<label>/
           registry.json                (parent entry; aggregated results)
           jobs/f<KK>_s<SEED>/          (one entry per fold-seed job)
             job.json  DONE             (crash-safe resume marker)
             best_model/ run_info.json  (train() artifacts)
           seed_<SEED>_summary.csv      (per-fold test metrics)
           seed_<SEED>_stitched.csv     (stitched OOS curve, equity+MTM)
           baseline_report.json         (aggregate + diagnostics)

Usage:
  .venv/bin/python run_baseline.py --label baseline-3seed --seeds 42,43,44
  .venv/bin/python run_baseline.py --resume runs/<dir>          # crash resume
  .venv/bin/python run_baseline.py --resume runs/<dir> --seeds 45,46   # extension:
  #   trains ONLY the new seeds (existing seeds keep their DONE markers) but
  #   aggregation always finalizes over ALL seeds present on disk in <dir>.
  (internal) --job K SEED --run-dir DIR                          # one job
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

RATIFIED_SEED_SET = [42, 43, 44, 45, 46]
STEPS_DEFAULT = 3_000_000


# ── shared helpers ───────────────────────────────────────────────────────────

def _load_folds(folds_limit: int | None = None):
    from config import CFG
    import train_ppo
    from data_loader import make_sliding_folds

    m1, feat, cols = train_ppo._load_decision_features()
    folds = make_sliding_folds(
        feat, train_years=CFG.sliding_train_years, val_months=CFG.sliding_val_months,
        test_months=CFG.sliding_test_months, step_months=CFG.sliding_step_months,
        embargo_bars=CFG.split_embargo_bars, lockbox_start=CFG.lockbox_start_date)
    if folds_limit:
        folds = folds[-folds_limit:]
    return m1, cols, folds


def _stitch(equities):
    """Chain per-fold test equity curves (realized + MTM) — mirrors the
    train_sliding_walk_forward stitch (duplicated on purpose: the launcher
    must not modify the system under test mid-anchor)."""
    import pandas as pd
    from config import CFG

    running, parts = CFG.initial_equity, []
    for eq in equities:
        if eq is None or eq.empty or "equity" not in eq:
            continue
        factor = running / CFG.initial_equity
        block = pd.DataFrame({"equity": eq["equity"].astype(float) * factor})
        if "equity_mtm" in eq.columns:
            block["equity_mtm"] = eq["equity_mtm"].astype(float) * factor
        parts.append(block)
        running = float(block["equity"].iloc[-1])
    import pandas as pd  # noqa: F811
    return pd.concat(parts) if parts else pd.DataFrame(columns=["equity"])


# ── one fold-seed job (runs in its own subprocess) ───────────────────────────

def run_job(k: int, seed: int, run_dir: Path, steps: int, folds_limit: int | None) -> None:
    import pandas as pd
    from config import CFG
    import train_ppo

    job_dir = run_dir / "jobs" / f"f{k:02d}_s{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    m1, cols, folds = _load_folds(folds_limit)
    tr, va, te = folds[k - 1]

    job = {"fold": k, "seed": seed, "status": "running",
           "train": f"{tr.index.min().date()}->{tr.index.max().date()}",
           "test": f"{te.index.min().date()}->{te.index.max().date()}"}
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))

    train_ppo.train(
        total_timesteps=steps, seed=seed, out_dir=str(job_dir),
        train_episode_steps=2048, eval_freq=max(25_000, steps // 20),
        dd_penalty=0.8, n_envs=4, device="auto", reveal_test=False,
        datasets=(m1, cols, tr, va, te))

    model, vp = train_ppo._load_fold_model(str(job_dir))
    test_eq, test_trades, test_rep = train_ppo._rollout_on_split(model, vp, m1, cols, te)
    test_eq.to_csv(job_dir / "test_equity.csv")
    test_trades.to_csv(job_dir / "test_trades.csv", index=False)

    run_info = json.loads((job_dir / "run_info.json").read_text())
    evals_path = job_dir / "eval_logs" / "consistency_evals.csv"
    evals = (pd.read_csv(evals_path) if evals_path.exists() else
             pd.DataFrame(columns=["train_eval_r", "val_r", "eligible"]))
    exit_mix = (test_trades["exit_reason"].value_counts().to_dict()
                if len(test_trades) else {})
    job.update({
        "status": "done",
        "wallclock_s": round(time.monotonic() - t0),
        "has_eligible_checkpoint": "best_model_path" in run_info,
        "n_evals": int(len(evals)),
        "n_eligible_evals": int(evals["eligible"].sum()),
        "eval_sign_counts": {   # train-leg / val-leg sign pattern per eval
            "train+val+": int(((evals.train_eval_r > 0) & (evals.val_r > 0)).sum()),
            "train+val-": int(((evals.train_eval_r > 0) & (evals.val_r <= 0)).sum()),
            "train-val+": int(((evals.train_eval_r <= 0) & (evals.val_r > 0)).sum()),
            "train-val-": int(((evals.train_eval_r <= 0) & (evals.val_r <= 0)).sum()),
        },
        "test_report": {kk: round(float(vv), 6)
                        for kk, vv in test_rep.items()
                        if isinstance(vv, (int, float)) and vv == vv},
        "exit_reason_mix": exit_mix,
    })
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    (job_dir / "DONE").write_text("ok\n")
    print(f"[job f{k:02d}_s{seed}] done in {job['wallclock_s']}s", flush=True)


# ── the pool manager ─────────────────────────────────────────────────────────

def manage(run_dir: Path, seeds: list[int], steps: int, concurrency: int,
           folds_limit: int | None) -> None:
    _, _, folds = _load_folds(folds_limit)
    n_folds = len(folds)
    jobs = [(k, s) for s in seeds for k in range(1, n_folds + 1)]
    pending = [(k, s) for (k, s) in jobs
               if not (run_dir / "jobs" / f"f{k:02d}_s{s}" / "DONE").exists()]
    print(f"[pool] {len(jobs)} jobs ({n_folds} folds x seeds {seeds}); "
          f"{len(jobs) - len(pending)} already done; {len(pending)} to run; "
          f"concurrency {concurrency}", flush=True)

    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    active: dict[tuple, subprocess.Popen] = {}
    failed: list[tuple] = []
    t0 = time.monotonic()
    while pending or active:
        while pending and len(active) < concurrency:
            k, s = pending.pop(0)
            cmd = [sys.executable, __file__, "--job", str(k), str(s),
                   "--run-dir", str(run_dir), "--steps", str(steps)]
            if folds_limit:
                cmd += ["--folds-limit", str(folds_limit)]
            (run_dir / "jobs").mkdir(exist_ok=True)
            log = open(run_dir / "jobs" / f"pool_f{k:02d}_s{s}.log", "w")
            active[(k, s)] = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
        time.sleep(10)
        for key, proc in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del active[key]
            k, s = key
            if rc != 0 or not (run_dir / "jobs" / f"f{k:02d}_s{s}" / "DONE").exists():
                failed.append(key)
                print(f"[pool] JOB FAILED f{k:02d}_s{s} (rc={rc}) — resume will retry",
                      flush=True)
            else:
                done = len(jobs) - len(pending) - len(active) - len(failed)
                el = (time.monotonic() - t0) / 3600
                print(f"[pool] {done}/{len(jobs)} done  ({el:.1f}h elapsed)", flush=True)
    if failed:
        print(f"[pool] {len(failed)} FAILED jobs: {failed} — re-run with --resume",
              flush=True)
        raise SystemExit(3)


# ── aggregation + diagnostics ────────────────────────────────────────────────

def aggregate(run_dir: Path, seeds: list[int], folds_limit: int | None,
              provisional: bool, candidate: bool = False) -> None:
    import numpy as np
    import pandas as pd
    from config import CFG
    import train_ppo
    from eval_harness import metric_return_over_mtm_dd
    from evaluate import full_report
    from run_registry import finalize_run, set_running_best

    _, _, folds = _load_folds(folds_limit)
    n_folds = len(folds)

    per_seed, diag_jobs = {}, []
    for s in seeds:
        rows, equities, trade_frames = [], [], []
        for k in range(1, n_folds + 1):
            jd = run_dir / "jobs" / f"f{k:02d}_s{s}"
            job = json.loads((jd / "job.json").read_text())
            assert job["status"] == "done", f"job f{k:02d}_s{s} not done"
            diag_jobs.append(job)
            eq = pd.read_csv(jd / "test_equity.csv", index_col=0)
            trades = pd.read_csv(jd / "test_trades.csv") if \
                (jd / "test_trades.csv").stat().st_size > 5 else pd.DataFrame()
            equities.append(eq)
            if len(trades):
                trade_frames.append(trades)
            rep = job["test_report"]
            rows.append({
                "fold": k, "seed": s, "test": job["test"],
                "test_return_pct": rep.get("total_return_pct"),
                "test_profit_factor": rep.get("profit_factor"),
                "test_sharpe_trade": rep.get("sharpe_trade"),
                "test_max_dd_mtm_pct": rep.get("max_drawdown_mtm_pct"),
                "test_n_trades": rep.get("n_trades"),
                "fold_metric": metric_return_over_mtm_dd(eq, CFG.initial_equity),
                "has_eligible_checkpoint": job["has_eligible_checkpoint"],
            })
        summary = pd.DataFrame(rows)
        summary.to_csv(run_dir / f"seed_{s}_summary.csv", index=False)
        stitched = _stitch(equities)
        stitched.to_csv(run_dir / f"seed_{s}_stitched.csv")
        all_trades = (pd.concat(trade_frames, ignore_index=True)
                      if trade_frames else pd.DataFrame())
        oos = full_report(stitched, all_trades, initial_equity=CFG.initial_equity,
                          periods_per_year=CFG.periods_per_year)["value"].to_dict()
        passed, detail = train_ppo._passes_consistency_gate(
            summary, ret_col="test_return_pct", pf_col="test_profit_factor",
            sharpe_col="test_sharpe_trade")
        per_seed[s] = {
            "metric_stitched": metric_return_over_mtm_dd(stitched, CFG.initial_equity),
            "stitched_return_pct": oos.get("total_return_pct"),
            "stitched_max_dd_mtm_pct": oos.get("max_drawdown_mtm_pct"),
            "stitched_ulcer_mtm": oos.get("ulcer_index_mtm"),
            "stitched_pf": oos.get("profit_factor"),
            "stitched_sharpe_trade": oos.get("sharpe_trade"),
            "n_trades": oos.get("n_trades"),
            "gate_passed": bool(passed),
            "gate_detail": detail,
            "folds_positive": int((summary.test_return_pct > 0).sum()),
            "folds_with_eligible": int(summary.has_eligible_checkpoint.sum()),
            "fold_metrics": summary.fold_metric.round(4).tolist(),
        }

    metrics = [per_seed[s]["metric_stitched"] for s in seeds]
    med = float(np.median(metrics))
    iqr = float(np.percentile(metrics, 75) - np.percentile(metrics, 25))
    sign_tot = {k: 0 for k in ("train+val+", "train+val-", "train-val+", "train-val-")}
    exit_tot: dict[str, int] = {}
    for j in diag_jobs:
        for k, v in j["eval_sign_counts"].items():
            sign_tot[k] += v
        for k, v in j.get("exit_reason_mix", {}).items():
            exit_tot[k] = exit_tot.get(k, 0) + v
    n_jobs = len(diag_jobs)
    report = {
        "provisional": provisional,
        "seeds": seeds,
        "n_folds": n_folds,
        "metric_per_seed": {str(s): round(per_seed[s]["metric_stitched"], 4) for s in seeds},
        "metric_median": round(med, 4),
        "metric_seed_iqr": round(iqr, 4),
        "per_seed": {str(s): {k: v for k, v in per_seed[s].items() if k != "fold_metrics"}
                     for s in seeds},
        "fold_metrics_per_seed": {str(s): per_seed[s]["fold_metrics"] for s in seeds},
        "eligibility": {
            "jobs_with_eligible_checkpoint":
                sum(j["has_eligible_checkpoint"] for j in diag_jobs),
            "n_jobs": n_jobs,
            "eligible_eval_fraction":
                round(sum(j["n_eligible_evals"] for j in diag_jobs)
                      / max(sum(j["n_evals"] for j in diag_jobs), 1), 4),
        },
        "eval_sign_counts_total": sign_tot,
        "exit_reason_mix_total": exit_tot,
        "wallclock_total_s": sum(j.get("wallclock_s", 0) for j in diag_jobs),
    }
    (run_dir / "baseline_report.json").write_text(json.dumps(report, indent=2))

    gates = [per_seed[s]["gate_passed"] for s in seeds]
    tag = "PROVISIONAL median-of-3" if provisional else "ratified median-of-5"
    if candidate:
        # Phase-C variant: record the run + INDEX line but NEVER move running-best
        # (the pointer moves only via the ratified ship rule — see ab_report.py).
        from config import CFG as _CFG
        finalize_run(run_dir, metric_median=med, gate_passed=all(gates),
                     verdict=(f"Phase-C CANDIDATE turnover_penalty_r={_CFG.turnover_penalty_r} "
                              f"({tag}); gates {sum(gates)}/{len(gates)}; running-best UNCHANGED "
                              f"(ship decided by ab_report.py)"), extra=report)
        print(json.dumps(report, indent=2))
        print("[aggregate] CANDIDATE report written; INDEX appended; running-best "
              "DELIBERATELY NOT moved (Phase-C ship rule decides).", flush=True)
    else:
        finalize_run(run_dir, metric_median=med, gate_passed=all(gates),
                     verdict=f"BASELINE anchor ({tag}); gates {sum(gates)}/{len(gates)}",
                     extra=report)
        set_running_best(run_dir, f"baseline anchor, metric median {med:+.4f} ({tag})")
        print(json.dumps(report, indent=2))
        print("[aggregate] baseline_report.json written; INDEX + running-best updated",
              flush=True)


# ── entry ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline-3seed")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--folds-limit", type=int, default=None)
    ap.add_argument("--resume", default=None, help="existing parent run dir")
    ap.add_argument("--job", nargs=2, type=int, default=None, metavar=("FOLD", "SEED"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--candidate", action="store_true",
                    help="Phase-C variant run: record + INDEX line, but do NOT move "
                         "the running-best pointer (ship rule via ab_report.py decides).")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.job:  # internal single-job mode
        run_job(args.job[0], args.job[1], Path(args.run_dir), args.steps,
                args.folds_limit)
        return

    # Phase-C guardrail (fail-fast): ANY non-anchor knob (turnover penalty, cost
    # randomization, …) MUST be a candidate run, else aggregate() would move the
    # running-best pointer for a variant. Placed AFTER the --job early-return so worker
    # subprocesses (knob set, no --candidate) are unaffected — only the launcher is gated.
    from config import CFG as _CFG
    _nonanchor = {k: v for k, v, anchor_v in
                  (("turnover_penalty_r", _CFG.turnover_penalty_r, 0.0),
                   ("cost_rand_frac", _CFG.cost_rand_frac, 0.0),
                   ("sliding_train_years", _CFG.sliding_train_years, 5.0))
                  if v != anchor_v}
    if _nonanchor and not args.candidate:
        raise SystemExit(
            f"Refusing to launch: non-anchor knob(s) {_nonanchor} without --candidate. A "
            f"non-anchor run must not move running-best — pass --candidate (Phase-C A/B) "
            f"or unset the knob(s).")

    if args.resume:
        run_dir = Path(args.resume)
        assert (run_dir / "registry.json").exists(), f"not a run dir: {run_dir}"
    else:
        from run_registry import new_run
        run_dir = new_run(args.label, seeds)
        reg_path = run_dir / "registry.json"
        reg = json.loads(reg_path.read_text())
        reg["ratified_seed_set"] = RATIFIED_SEED_SET          # pre-declared
        reg["provisional_seeds"] = seeds
        reg["anchor_discipline"] = (
            "No config/hparam/cost/eligibility/gate changes in response to this "
            "run. Observations -> Phase-C candidates list only.")
        from config import CFG as _CFG  # record the effective Phase-C A/B knobs
        reg["turnover_penalty_r"] = _CFG.turnover_penalty_r   # 0.0 = anchor
        reg["turnover_entry_frac"] = _CFG.turnover_entry_frac  # 1.0 = flat (A/B #1)
        reg["cost_rand_frac"] = _CFG.cost_rand_frac           # 0.0 = anchor
        reg["sliding_train_years"] = _CFG.sliding_train_years  # 5.0 = anchor (A/B #4: 10.0)
        reg_path.write_text(json.dumps(reg, indent=2))
        print(f"[pool] parent run: {run_dir}", flush=True)

    if not args.aggregate_only:
        manage(run_dir, seeds, args.steps, args.concurrency, args.folds_limit)

    # Aggregate over EVERY seed present on disk in the parent run — NOT just this
    # invocation's --seeds. An extension (--resume --seeds 45,46) trains only the
    # new seeds but must finalize the FULL baseline; deriving the set from disk
    # makes "median-of-2 mislabeled as the anchor" structurally impossible.
    _, _, _folds = _load_folds(args.folds_limit)
    from collections import Counter
    done = Counter(int(p.name.split("_s")[1])
                   for p in (run_dir / "jobs").glob("f*_s*")
                   if (p / "DONE").exists())
    agg_seeds = sorted(s for s, c in done.items() if c == len(_folds))
    print(f"[aggregate] seeds complete on disk: {agg_seeds}", flush=True)
    aggregate(run_dir, agg_seeds, args.folds_limit,
              provisional=(set(agg_seeds) != set(RATIFIED_SEED_SET)),
              candidate=args.candidate)


if __name__ == "__main__":
    main()
