"""Repro/check for the B3 fix (doc 01): test_analysis.ipynb must evaluate the
best checkpoint under ITS OWN VecNormalize snapshot (best_model_vecnorm.pkl),
not the final model's VECNORM_PATH. test_analysis_folds.ipynb is the reference
implementation that always did this correctly.

Static JSON checks (no kernel execution needed):
  1. test_analysis.ipynb defines BEST_VECNORM_PATH from best_model_vecnorm_path.
  2. Every run_model_on_split(BEST_VAL_MODEL, ...) call passes BEST_VECNORM_PATH.
  3. The reference invariant still holds in test_analysis_folds.ipynb.

Run:  .venv/bin/python checks/check_b3_notebook_pairing.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def check(nb_name: str) -> None:
    src = _code(ROOT / "notebooks" / nb_name)

    assert "BEST_VECNORM_PATH" in src, f"{nb_name}: BEST_VECNORM_PATH not defined"
    assert "best_model_vecnorm_path" in src, \
        f"{nb_name}: BEST_VECNORM_PATH not sourced from run_info's best_model_vecnorm_path"

    calls = re.findall(r"run_model_on_split\(\s*(\w+)\s*,\s*(\w+)\s*,", src)
    assert calls, f"{nb_name}: no run_model_on_split calls found"
    bv_calls = [(m, v) for m, v in calls if m == "BEST_VAL_MODEL"]
    assert bv_calls, f"{nb_name}: no BEST_VAL_MODEL evaluations found"
    for model_arg, vec_arg in bv_calls:
        assert vec_arg == "BEST_VECNORM_PATH", (
            f"{nb_name}: BEST_VAL_MODEL paired with {vec_arg} (must be BEST_VECNORM_PATH)")
    # The final model must still use its own (final) vecnorm.
    final_calls = [(m, v) for m, v in calls if m == "FINAL_MODEL"]
    for model_arg, vec_arg in final_calls:
        assert vec_arg == "VECNORM_PATH", (
            f"{nb_name}: FINAL_MODEL paired with {vec_arg} (must be VECNORM_PATH)")
    print(f"  {nb_name}: best↔best vecnorm, final↔final vecnorm "
          f"({len(bv_calls)} best-val calls, {len(final_calls)} final calls): PASS")


if __name__ == "__main__":
    print("B3 notebook-pairing checks:")
    check("test_analysis.ipynb")        # the fixed notebook
    check("test_analysis_folds.ipynb")  # the reference (must keep holding)
    print("B3 notebook-pairing checks: ALL PASS")
