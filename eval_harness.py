"""Multi-seed evaluation harness (doc 04 Phase A).

Every later A/B comparison calls this: run one variant under several seeds,
collect a scalar metric per seed plus the gate verdict, and return the
DISTRIBUTION (median + spread) — never a single-seed point estimate.

The harness is metric-agnostic: `run_fn(seed)` does whatever a variant needs
(train, roll out, read artifacts) and returns a dict with at least
`{"metric": <float>}` and optionally `"gate_passed"` (True/False/None as in
run_info). `metric_return_over_mtm_dd` below is the PROVISIONAL default metric
helper — the primary metric + ship threshold are pinned in doc 04 Phase A task
5 and must be ratified before any ship/kill decision uses them.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from evaluate import drawdown

# The doc-04 §5 open decision "seeds per variant: 3 or 5" — cheap ranking uses
# the first 3, finalist runs use 5 (doc 04 Phase C compute guardrail).
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def metric_return_over_mtm_dd(equity_df: pd.DataFrame,
                              initial_equity: float = 10_000.0) -> float:
    """PROVISIONAL metric helper: total return ÷ |max MTM drawdown|.

    Uses the mark-to-market curve when present (honest DD); NaN on a flat/
    empty curve (a do-nothing policy has no meaningful risk-adjusted return —
    the gate's min-trades guard handles rejection).
    """
    if equity_df is None or equity_df.empty or "equity" not in equity_df:
        return float("nan")
    eq = equity_df["equity"].astype(float)
    total_return = float(eq.iloc[-1]) / float(initial_equity) - 1.0
    col = "equity_mtm" if "equity_mtm" in equity_df.columns else "equity"
    max_dd = abs(float(drawdown(equity_df[col].astype(float)).min()))
    if not np.isfinite(max_dd) or max_dd < 1e-9:
        return float("nan")
    return total_return / max_dd


def multi_seed_run(run_fn: Callable[[int], dict],
                   seeds: Iterable[int] = DEFAULT_SEEDS[:3],
                   metric_key: str = "metric") -> dict:
    """Run `run_fn` once per seed and aggregate the metric distribution.

    Returns dict with: per_seed (DataFrame indexed by seed), n_seeds, median,
    q25/q75/iqr, min/max, and n_gate_failed. Gate failures do NOT drop a seed
    from the distribution — a failed gate is evidence about the variant, and
    hiding it would bias the comparison — but they are counted and reported
    loudly so no ship decision can miss them.
    """
    rows = []
    for seed in seeds:
        out = dict(run_fn(int(seed)))
        if metric_key not in out:
            raise KeyError(f"run_fn(seed={seed}) returned no '{metric_key}'")
        out["seed"] = int(seed)
        rows.append(out)

    per_seed = pd.DataFrame(rows).set_index("seed")
    m = per_seed[metric_key].astype(float)
    gates = per_seed["gate_passed"] if "gate_passed" in per_seed else None
    n_gate_failed = int((gates == False).sum()) if gates is not None else 0  # noqa: E712

    result = {
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "median": float(m.median()),
        "q25": float(m.quantile(0.25)),
        "q75": float(m.quantile(0.75)),
        "iqr": float(m.quantile(0.75) - m.quantile(0.25)),
        "min": float(m.min()),
        "max": float(m.max()),
        "n_gate_failed": n_gate_failed,
    }
    if n_gate_failed:
        banner = "!" * 74
        print(f"\n{banner}\n  WARNING: {n_gate_failed}/{len(per_seed)} seed(s) "
              f"FAILED the deployment gate — this variant is not shippable\n"
              f"  regardless of its metric distribution.\n{banner}\n")
    return result


def describe(result: dict, label: str = "variant") -> str:
    """One-line human summary of a multi_seed_run result."""
    return (f"{label}: median={result['median']:+.3f}  "
            f"IQR=[{result['q25']:+.3f}, {result['q75']:+.3f}]  "
            f"range=[{result['min']:+.3f}, {result['max']:+.3f}]  "
            f"seeds={result['n_seeds']}  gate_failed={result['n_gate_failed']}")
