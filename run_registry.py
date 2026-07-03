"""Append-only run registry (Phase B+, ratified 2026-07-02).

Every full training run writes into its own versioned directory —
``runs/<UTCdate>_<gitsha7>_<label>/`` — so results are permanently traceable
to the exact code (git SHA), data (file name + SHA256 + span), and config
(full CFG snapshot). ``runs/`` is gitignored EXCEPT ``runs/INDEX.md``: one
committed line per run (date, sha, label, metric median, gate, verdict).

Disciplines encoded here:
- The "running best" is a pointer line in INDEX.md and moves ONLY when the
  ratified ship rule passes (>= +10% relative median AND >= 4/5 seeds beat the
  running-best median AND no gate regression). Never edit it for a near-miss.
- ``models/`` remains the production slot for the promoted pair; the registry
  is the experiment record, not the deployment mechanism.

Usage:
    run_dir = new_run("baseline", seeds=[42, 43, 44, 45, 46])
    ... train into run_dir (per-fold artifacts, stitched OOS curve) ...
    finalize_run(run_dir, metric_median=0.87, gate_passed=True,
                 verdict="baseline anchor")
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
INDEX = RUNS_DIR / "INDEX.md"


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _data_fingerprint(csv_path: Path) -> dict[str, Any]:
    """Name + SHA256 + size + first/last timestamp of the data file."""
    csv_path = Path(csv_path)
    fp: dict[str, Any] = {"name": csv_path.name,
                          "size_bytes": csv_path.stat().st_size,
                          "sha256": _sha256(csv_path)}
    with open(csv_path, "rb") as f:
        header = f.readline()
        first = f.readline().decode(errors="replace").split(",")[0]
        f.seek(max(csv_path.stat().st_size - 4096, 0))
        last = f.read().decode(errors="replace").strip().splitlines()[-1].split(",")[0]
    fp["span"] = f"{first} -> {last}"
    _ = header
    return fp


def new_run(label: str, seeds: Iterable[int], base_dir: Path | None = None) -> Path:
    """Create the run directory + registry.json; returns the run dir."""
    from config import CFG

    base = Path(base_dir) if base_dir else RUNS_DIR
    # Second-resolution stamp: same-day runs (multi-seed sweeps, retries after
    # a crash) must never collide — the registry is append-only, never reused.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_{_git_sha()}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = dataclasses.asdict(CFG)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    registry = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "seeds": list(seeds),
        "data": _data_fingerprint(Path(CFG.csv_path)),
        "cfg": cfg,
        "results": None,   # filled by finalize_run
    }
    (run_dir / "registry.json").write_text(json.dumps(registry, indent=2))
    return run_dir


def finalize_run(run_dir: Path, metric_median: float | None = None,
                 gate_passed: bool | None = None, verdict: str = "",
                 extra: dict[str, Any] | None = None,
                 index_path: Path | None = None) -> None:
    """Record results into registry.json and append one INDEX.md line."""
    run_dir = Path(run_dir)
    reg_path = run_dir / "registry.json"
    registry = json.loads(reg_path.read_text())
    registry["results"] = {
        "metric_median": metric_median,
        "gate_passed": gate_passed,
        "verdict": verdict,
        "finalized_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    reg_path.write_text(json.dumps(registry, indent=2))

    idx = Path(index_path) if index_path else INDEX
    idx.parent.mkdir(parents=True, exist_ok=True)
    if not idx.exists():
        idx.write_text(
            "# Run index\n\n"
            "Append-only. One line per full training run (see run_registry.py).\n\n"
            "**Running best:** (none yet — moves ONLY via the ratified ship rule: "
            ">= +10% relative median AND >= 4/5 seeds beat the running-best median "
            "AND no gate regression.)\n\n"
            "| date (UTC) | git sha | label | metric median | gate | verdict |\n"
            "|---|---|---|---|---|---|\n")
    metric_s = "—" if metric_median is None else f"{metric_median:+.4f}"
    gate_s = {True: "PASS", False: "FAIL", None: "—"}[gate_passed]
    line = (f"| {registry['created_utc'][:10]} | {registry['git_sha']} | "
            f"{registry['label']} | {metric_s} | {gate_s} | {verdict} |\n")
    with open(idx, "a") as f:
        f.write(line)
