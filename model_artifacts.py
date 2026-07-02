from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a project-relative artifact path without forcing it to exist."""
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def resolve_sb3_model_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve an SB3 model path across both `.zip` and legacy no-extension saves.

    Legacy checkpoints in this project were saved without an explicit `.zip`
    suffix because the slug contains decimal points, so `Path.suffix` was not
    empty and SB3 did not auto-append `.zip` on write.
    """
    requested = resolve_project_path(path, base_dir=base_dir)
    candidates = [requested]

    requested_str = str(requested)
    if requested_str.lower().endswith(".zip"):
        candidates.append(Path(requested_str[:-4]))
    else:
        candidates.append(Path(f"{requested}.zip"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Stable-Baselines3 model artifact not found. Tried:\n{tried}")


def load_run_info(models_dir: str | Path = "models") -> tuple[Path, dict[str, Any]]:
    models_path = Path(models_dir)
    info_path = models_path / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found.\nRun train_ppo.py first, then retry."
        )
    return info_path, json.loads(info_path.read_text())


def check_gate_approval(run_info: dict[str, Any], allow_failed: bool = False,
                        context: str = "this evaluation") -> None:
    """B1 guard: refuse to run reporting/holdout tooling on a gate-FAILED model.

    The walk-forward deployment gate records its verdict in
    ``run_info["gate_passed"]`` (train_ppo._promote_fold_to_production), but
    before this guard nothing downstream ever read it — a gate-failed model was
    silently revealed on the sealed holdout and reported (doc 01, bug B1).

    Semantics:
    - key absent      → no gate ran (e.g. single-split train()); proceed quietly.
    - gate_passed=True  → proceed quietly.
    - gate_passed=False → print a loud banner and raise SystemExit(2), unless
      ``allow_failed=True`` — then proceed with a loud warning. The override is
      explicit, never silent: a failed model stays inspectable, but every
      consumer must knowingly opt in.
    """
    gate = run_info.get("gate_passed")
    if gate is None or bool(gate):
        return
    banner = "!" * 74
    if allow_failed:
        print(f"\n{banner}")
        print("  WARNING: gate_passed=false — this model FAILED the walk-forward")
        print(f"  consistency gate and is NOT approved for deployment. Proceeding with")
        print(f"  {context} because --allow-failed-gate was explicitly enabled.")
        print("  Treat every number that follows as NOT deployable.")
        print(f"{banner}\n")
        return
    print(f"\n{banner}")
    print("  BLOCKED: gate_passed=false — this model FAILED the walk-forward")
    print("  consistency gate (see NO_DEPLOY.txt next to run_info.json).")
    print(f"  Refusing {context} so a non-approved model cannot be")
    print("  silently revealed or reported (doc 01, bug B1).")
    print("  To inspect it anyway, opt in explicitly:")
    print("    --allow-failed-gate       (final_holdout_eval.py / training_diagnostics.py)")
    print("    ALLOW_FAILED_GATE = True  (notebooks/test_analysis.ipynb)")
    print(f"{banner}\n")
    raise SystemExit(2)
