"""Repro/check for the B1 fix (doc 01): the deploy gate must actually gate.

Verifies, at unit level and end-to-end, that reporting tools honor
run_info["gate_passed"]:
  1. check_gate_approval semantics (absent / true / false / false+override).
  2. final_holdout_eval.py BLOCKS (exit 2) on a fabricated gate-failed run_info
     and proceeds past the guard with --allow-failed-gate.
  3. Static: the notebook guard + training_diagnostics wiring exist.

Uses models/smoke/ artifacts as the fixture; writes a TEMPORARY
models/run_info.json and removes it afterwards (refuses to run if one exists).

Run:  .venv/bin/python checks/check_b1_gate_guard.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_artifacts import check_gate_approval  # noqa: E402


def unit_checks() -> None:
    # Absent key (single-split run, no gate) and passed gate → proceed quietly.
    check_gate_approval({}, allow_failed=False)
    check_gate_approval({"gate_passed": True}, allow_failed=False)
    # Failed gate + explicit override → proceed (loud warning).
    check_gate_approval({"gate_passed": False}, allow_failed=True)
    # Failed gate, no override → SystemExit(2).
    try:
        check_gate_approval({"gate_passed": False}, allow_failed=False)
    except SystemExit as e:
        assert e.code == 2, f"expected exit code 2, got {e.code}"
    else:
        raise AssertionError("guard did NOT fire on gate_passed=false")
    print("  [1] unit semantics: PASS")


def integration_check() -> None:
    smoke_info = ROOT / "models" / "smoke" / "run_info.json"
    assert smoke_info.exists(), "run smoke_test.py first (needs models/smoke fixtures)"
    fixture = ROOT / "models" / "run_info.json"
    assert not fixture.exists(), (
        f"{fixture} already exists — refusing to overwrite a real production "
        f"run_info. Move it aside and re-run."
    )
    info = json.loads(smoke_info.read_text())
    info["gate_passed"] = False
    fixture.write_text(json.dumps(info, indent=2))
    py = str(ROOT / ".venv" / "bin" / "python")
    try:
        r = subprocess.run([py, "final_holdout_eval.py"], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 2, (
            f"expected BLOCK exit 2, got {r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        assert "BLOCKED" in r.stdout and "gate_passed=false" in r.stdout
        print("  [2a] final_holdout_eval blocks on failed gate (exit 2): PASS")

        r2 = subprocess.run([py, "final_holdout_eval.py", "--allow-failed-gate"],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
        # Guard released → loud warning, then proceeds (on this machine it later
        # stops at the default CSV being absent — a DIFFERENT failure, which is
        # exactly the proof that the gate guard itself no longer blocks).
        assert "WARNING" in r2.stdout and "BLOCKED" not in r2.stdout
        assert r2.returncode != 2
        print("  [2b] --allow-failed-gate overrides with loud warning: PASS")
    finally:
        fixture.unlink(missing_ok=True)


def static_checks() -> None:
    nb = json.loads((ROOT / "notebooks" / "test_analysis.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "ALLOW_FAILED_GATE" in src and "gate_passed" in src, \
        "notebook guard missing from test_analysis.ipynb"
    td = (ROOT / "training_diagnostics.py").read_text()
    assert "check_gate_approval(" in td and "allow_failed_gate" in td, \
        "training_diagnostics.py guard wiring missing"
    print("  [3] notebook + diagnostics wiring present: PASS")


if __name__ == "__main__":
    print("B1 gate-guard checks:")
    unit_checks()
    integration_check()
    static_checks()
    print("B1 gate-guard checks: ALL PASS")
