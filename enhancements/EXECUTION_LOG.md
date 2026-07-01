# Execution log

Chronological record of what was actually run during execution of the doc-04 plan,
its results, and any deviation. One section per task. Facts are cited to `file:line`
or to the command/output that produced them.

---

## Task 0 — Environment + git snapshot  ✅ (2026-07-01)

**Machine:** darwin (arm64), Python **3.9.6** (Xcode system interpreter).

**RL stack:** was missing at session start (`gymnasium`, `stable_baselines3`,
`torch`, `plotly` all `ModuleNotFoundError`; only `numpy 2.0.2` / `pandas 2.3.3` /
`matplotlib 3.9.4` present) — matches the historical env note.

**Action:** created a project venv and installed the pinned-by-`>=` requirements:
```
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```
**Resolved versions (import-verified in the venv):**
python 3.9.6 · gymnasium **1.1.1** · stable-baselines3 **2.7.1** · torch **2.8.0** ·
numpy 2.0.2 · pandas 2.3.3 · plotly 6.8.0 · scikit-learn 1.6.1.

> ⚠️ **Watch-item for Task 1:** these are *newer* than the versions the repo code was
> written against (requirements use `>=`, so pip took the latest). gymnasium 1.x /
> sb3 2.7 / numpy 2.0 have known API drift vs. earlier releases. The Task-1 smoke-test
> is the first thing that will actually exercise this — treat any failure there as a
> possible version-compat issue, not necessarily a code bug. Consider capturing a
> `pip freeze` lockfile once the smoke-test passes, for reproducibility.

**Git snapshot:** branch `main`, single commit `7615a58 "commit 1"`, working tree
clean except `?? enhancements/` (untracked) — the whole plan corpus (01–04 + README)
is not yet version-controlled. `.venv/` is now also untracked (must be git-ignored).

**Task-1 data confirmed on disk:**
`../trading_bot/market_mechanics_bot/lean_migration/data/XAU_USD_M1.csv` — 69 MB,
**1,216,345** M1 rows (LEAN format `DateTime,Open,High,Low,Close,Volume`, UTC, mid).

**Checkpoint:** awaiting user OK on the branch/commit proposal before any code change.
