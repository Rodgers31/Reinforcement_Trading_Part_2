# Data sourcing & execution plan

**What this is:** the operational sequence we'll actually execute — how to get the
data, then the ordered build/test steps — tying together the analysis in
[`01-baseline-review-and-known-issues.md`](01-baseline-review-and-known-issues.md)
(bugs), [`02-multi-instrument-training-plan.md`](02-multi-instrument-training-plan.md)
(multi-instrument), and [`03-rl-core-enhancements.md`](03-rl-core-enhancements.md)
(RL-core enhancements). Planning artifact only — no code or training runs yet.

**The core principle (confirmed):** fix the **measurement bugs first — precisely
because they don't touch training** — so the baseline is measured with a correct
ruler; *then* baseline, *then* change one thing at a time and re-test. Fixing them
first doesn't contaminate the baseline (they don't alter the trained policy on the
default sliding path, doc 01); it just calibrates the scale before we weigh.

---

## 1. Data sourcing (do this first)

### 1.1 Requirements (verified against the loader)

**How far back** — set by the sliding walk-forward (5y train + 6m val + 6m test,
step 6m; `config.py:110-113`):

| History | Sliding folds | Verdict |
|---|---|---|
| ~3.5y (current OANDA data) | 0 | can't form one fold — single-split / smoke-test only |
| ~6y | 1 | toy |
| ~10–12y | ~8–12 | workable; real generalization signal |
| **~15–23y (target)** | ~18–34 | as designed (default file is 2003→2026) |

**Granularity:** decisions on **H1**; intrabar TP/SL fills need an execution stream —
**M1** ideal, **M5** acceptable via `execution_timeframe="5min"` (the env assumes
SL-first when both hit one bar, so coarser = slightly more pessimistic,
`env_bracket.py:218`). The execution stream must span the **same range** as the
decision history.

**Format** — the loader is flexible (`data_loader.py:28`): a CSV with O/H/L/C/V + a
time column. The existing LEAN CSV (`DateTime,Open,High,Low,Close,Volume`, UTC, mid)
works with three settings:
- `time_col="DateTime"` — required; the autodetect only matches columns starting
  with `"time"`, and `"DateTime"` starts with `"date"` (`data_loader.py:55`).
- `source_tz="UTC"` — sessions are computed in UTC internally (`features.py:37`), so
  they stay correct as long as this matches the file's clock.
- `timestamp_is_bar_open=True` — default; OANDA and Dukascopy both stamp bar-open.

**Price stream:** pick one, consistently. **Mid** (clean — spread modeled separately
via `spread_price`) or **Bid** (the RL default). Use OANDA's bid/ask to **measure the
real per-instrument spread** → feeds the doc-02 cost model and the `config.py:143`
calibration.

### 1.2 Sources — hybrid (verified on disk)

| Source | Role | Depth | Caveat |
|---|---|---|---|
| **Dukascopy** | deep training backbone | XAUUSD M1 ~2003→, FX majors, bid/ask | own downloader; the RL default file (`…2003.05.05…`) is Dukascopy-shaped |
| **OANDA (your bot)** | recent tail + real spreads + live parity + multi-instrument breadth | ~3.5y on disk (2023→2026); H1 ~2016 | too shallow for the backbone |

Verified this pass: `market_mechanics_bot/lean_migration/data/XAU_USD_M1.csv` =
2023-01-02 → 2026-06-11, 1.22M rows; the OANDA puller fetches **M5/M15/H1/H4** with
`price:"MBA"` (mid+bid+ask), **not M1** (`update_historical_data.py:89,424`); its own
comments cap OANDA depth at H1≈2016 / M15≈2y (`update_historical_data.py:181-187`).

**Why hybrid, not either/or.** OANDA is the *right* source for the recent tail, real
per-instrument spreads, the whole multi-instrument universe from one feed, and
eventual live parity (you'd trade on OANDA) — but it can't supply the ~20y backbone.
Dukascopy supplies the backbone. Because the model trains on **ATR-normalized**
features, small vendor-to-vendor price differences largely wash out — the splice is
low-risk, but verify it (step 3).

### 1.3 Acquisition steps
1. Pull ~20y XAUUSD **M1** (and the target FX pairs) from **Dukascopy**.
2. Convert to loader format (O/H/L/C/V + the three settings above); land in `data/`.
3. **Splice-check** against the OANDA overlap you already have (2023-2026): compare
   Dukascopy vs OANDA on the shared window — they must agree to within spread before
   stitching. Verify, don't assume. Include **volume comparability** in the same
   check (doc 05 N6): if vendor tick volumes don't reconcile, volume features are
   rejected permanently rather than silently unreliable.
4. Keep OANDA (your bot) for the recent tail + per-instrument spread calibration +
   eventual live parity.

### 1.4 Definition of done (data)
`data/XAUUSD_…M1….csv` spans ≥15y (target ~20y), loads cleanly through
`prepare_feature_frame`, passes the `leakage_checks.py` guardrail, and the
splice-check against OANDA agrees within spread.

---

## 2. Execution sequence

**Principle restated:** B1–B3 are **measurement/reporting bugs (the ruler)**, not
training bugs — they don't change the trained policy on the default sliding path
(doc 01). Fix them *before* the baseline, or the "before vs after fix" comparison
measures the ruler changing, not the strategy.

**Two distinct uses of "multi-seed" — don't conflate:**
- **Multi-seed *evaluation*** (run each variant under 3–5 seeds, compare
  distributions): a **protocol necessity**, part of the ruler (Phase A). One run per
  variant can't separate signal from seed noise (current code = single seed 42,
  `train_ppo.py:311`).
- **Multi-seed *ensembling*** (deploy a blend as a portfolio): an **enhancement**
  (doc 03 §3.7b), A/B'd like any other in Phase C.

### Phase A — Build the ruler (before any baseline)
- Fix **B1–B3** (doc 01): the gate must actually gate (holdout/diagnostics honor
  `gate_passed`); pair the best checkpoint with its **own** vecnorm (×2). Measurement
  only — no training change.
- Make risk metrics honest: **mark-to-market equity** + a trade-based/scaled Sharpe
  (doc 03 §3.9a/b), so drawdown and selection aren't understated. Do this in Phase A
  so every later comparison shares one honest ruler.
- Wire **multi-seed evaluation** (3–5 seeds); report distributions.
- **PINNED metric + ship threshold (RATIFIED 2026-07-01).** Primary = median-of-5
  of (stitched-OOS return ÷ |stitched-OOS max MTM DD|); ALSO report a path-based
  DD (Ulcer / return-over-avg-DD) as a non-gating secondary; ship rule = ≥+10%
  relative AND ≥4/5 seeds beat running-best median AND no gate regression;
  3 seeds rank / 5 finalists. The +10% margin is PROVISIONAL — reserve ONE
  recalibration against the baseline's measured seed-IQR, once, before any
  Phase-C A/B (calibrating to noise, not outcome). Implemented: `sharpe_trade`,
  `max_drawdown_mtm_pct`, `ulcer_index_mtm` in `evaluate.py`; harness in
  `eval_harness.py`.
- **PINNED gate re-form (doc 05 N3, RATIFIED 2026-07-01).** Fraction/quantile
  form, fold-count-invariant: breadth `min_consistent_fold_frac = 0.70`
  (return>0 AND PF>1; ceil(0.70×5)=4 reproduces the old block-scheme 4-of-5),
  floor = 10th-percentile fold PF ≥ 0.90, third leg = mean **trade-based**
  Sharpe > 0. Pinned by reasoning before any baseline existed. Gate: a sound
  gate rejecting the baseline is a RESULT, not a trigger to loosen; re-pin only
  for mechanical mis-specification, never to make a result pass.
- **Decide the fill-model fix** (doc 05 N1, gap-through-SL — measured at ~monthly
  >1×ATR gaps): recommended to land HERE so the baseline is measured under honest
  fills; deferring it to Phase C keeps a known-flattering anchor.
- **Carve the final lockbox** — a tail period excluded from the *entire* sweep (a
  walk-forward over all history leaves nothing untouched unless carved out now).
- **DoD:** a reproducible harness that, given a model, returns a multi-seed OOS
  distribution on the dev surface, honors `gate_passed`, and never touches the lockbox.

### Phase B — Baseline
- Current strategy **as-is** (no enhancements), multi-seed, on the dev-OOS surface,
  under the honest ruler (including the N1 fill model, if adopted in Phase A). This
  is the anchor every change is measured against.
- Optional: run the pre-fix "uncalibrated" numbers once and file them clearly
  labeled — for the record, not as the anchor.
- **DoD:** a logged baseline distribution + the pinned metric value.

### Phase C — Enhancements, one at a time
- Order per **doc 03 §6** (quick wins → high-leverage bets → research long-shots).
- Doc-05 additions queue here: the **N2+N4 env-correctness bundle** (zero-obs
  truncation bootstrap + terminated/truncated semantics, one attribution run);
  **representation batches** per doc 05 §2.5 (P1 range-position + P4 gap-awareness →
  P3 M1-microstructure → P2 observable cost, N6 volume if its data gate passes);
  the **P5 retrain-cadence sweep** once warm-start makes it affordable.
- Each substantive change: multi-seed, A/B vs the **running best**, ship/kill on the
  pre-committed threshold, one row in the decision log. Trivial hygiene may bundle;
  load-bearing changes go solo (attribution).
- **Compute:** rank candidates on a cheap proxy (fewer folds / fewer steps); reserve
  full multi-seed runs for finalists (a full sliding run ≈ 100M env-steps, doc 03 §3.6).
- **DoD (per change):** a decision-log row with the OOS delta distribution + ship/kill.

### Phase D — Multi-instrument
- Follow **doc 02** phasing (Phase 0 similarity diagnostic → per-pair specialists →
  pooled generalist + instrument tag → portfolio).
- Deployment design gains the **portfolio risk overlay** (doc 05 N7: aggregate-DD
  kill-switch, per-currency net-exposure caps, max concurrent books); config gains
  per-instrument `bars_per_year` (doc 05 N8); the VecNormalize **norm-mask**
  (doc 05 P7) is built once, together with the instrument tag.
- Yardstick **shifts**: not "better than gold-only on gold," but "does the
  portfolio's risk-adjusted return beat the single-instrument books."
- **DoD:** doc-02 go/no-go experiment resolved on the dev surface.

### Phase E — Final holdout reveal (once)
- Run the single final chosen system on the carved-out lockbox, **once**. That is the
  honest OOS number you report. If it disappoints, that's information — do not iterate
  against it (that would burn the holdout, doc 03 §7).

---

## 3. Decision log

One row per A/B: change · seeds · dev-OOS delta (median + spread) · compute ·
ship/kill · notes. Keeps the multiple-testing auditable (doc 03 §7) and makes it
visible which changes actually earned their place.

## 4. Guardrails (carried from doc 03 §7)
- Multi-seed always; compare distributions, not point estimates.
- Never bundle changes you want to attribute.
- Iterate on the dev surface; reveal the lockbox once.
- Pre-commit the metric + threshold; discount marginal single-idea OOS wins.
- Cheap proxy to rank; full runs for finalists.

## 5. Open decisions before starting
- Deep-history vendor: **Dukascopy** (default) or a paid vendor?
- Price stream: **mid** or **bid**?
- Execution timeframe: **M1** (needs deep M1) or **M5** fallback?
- Seeds per variant (drives compute): 3 or 5?
- Phase-D universe: which pairs, how many?
- N1 sequencing (doc 05): honest gap-fills before the baseline (recommended), or an
  attributed Phase-C A/B?
- Economic-calendar source for P9 (doc 05): acquire during the data work, or drop?

## 6. Next action

Two tracks in parallel:
- **0a (today, no acquisition):** wire the existing `XAU_USD_M1.csv` into `config.py`
  (`time_col="DateTime"`, `source_tz="UTC"`) and run a single-split smoke-test — this
  validates the pipeline **and** the Phase-A ruler end-to-end on real data you already
  have, decoupled from deep-history acquisition.
- **0b:** start the **Dukascopy** XAUUSD M1 pull for the real backbone.

Then proceed Phase A → B → C. The smoke-test (0a) de-risks all the pipeline/ruler
work so it isn't blocked on data acquisition (0b).
