"""Repro/check for the run registry (Phase B infra).

Creates a self-test run in a TEMP base dir (never pollutes runs/INDEX.md):
asserts the directory naming, the registry.json provenance fields (git sha,
data SHA256 + span, full CFG snapshot, seeds), and that finalize_run records
results and appends exactly one well-formed INDEX line.

Run:  .venv/bin/python checks/check_run_registry.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_registry import finalize_run, new_run  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        run_dir = new_run("registry-selftest", seeds=[42, 43, 44], base_dir=base)

        assert run_dir.exists() and run_dir.parent == base
        assert re.match(r"\d{8}_[0-9a-f]{7}(-dirty)?_registry-selftest$", run_dir.name), run_dir.name

        reg = json.loads((run_dir / "registry.json").read_text())
        assert reg["seeds"] == [42, 43, 44]
        assert re.match(r"[0-9a-f]{7}(-dirty)?$", reg["git_sha"])
        assert re.match(r"[0-9a-f]{64}$", reg["data"]["sha256"])
        assert "->" in reg["data"]["span"] and reg["data"]["size_bytes"] > 0
        assert reg["cfg"]["csv_path"] and "spread_atr_frac" in reg["cfg"]
        assert reg["results"] is None
        print("  [1] new_run: dir naming + provenance (sha/data-sha256/span/CFG/seeds): PASS")

        idx = base / "INDEX.md"
        finalize_run(run_dir, metric_median=0.1234, gate_passed=True,
                     verdict="selftest", index_path=idx)
        reg = json.loads((run_dir / "registry.json").read_text())
        assert reg["results"]["metric_median"] == 0.1234
        assert reg["results"]["gate_passed"] is True
        lines = idx.read_text().splitlines()
        rows = [l for l in lines if l.startswith("|") and "registry-selftest" in l]
        assert len(rows) == 1 and "+0.1234" in rows[0] and "PASS" in rows[0]
        assert any("Running best:" in l for l in lines)
        print("  [2] finalize_run: results recorded + one well-formed INDEX line: PASS")


if __name__ == "__main__":
    print("Run-registry checks:")
    main()
    print("Run-registry checks: ALL PASS")
