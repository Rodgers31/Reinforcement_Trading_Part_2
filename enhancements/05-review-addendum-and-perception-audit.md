# Skeptical-review addendum: missed items + perception audit

**What this is:** the output of a second, adversarial review pass (2026-07-01) done
with the RL stack **installed and running** (Task 0/1 of the execution log): every
doc-01 claim re-verified against live source, the installed `stable_baselines3
2.7.1` traced where env behavior depends on it, and a data census run on the real
OANDA/LEAN CSV. It records (a) defects/design gaps docs 01–04 **missed**, and
(b) a dedicated **perception & learning-flow audit** answering "does the agent see
enough to build a concrete model of what price is doing?"

Planning artifact only. Items here do NOT reorder the plan — they slot into the
existing doc-04 phases (merge table in §4). Doc-04's discipline applies unchanged:
one attributable change at a time, multi-seed, pre-committed thresholds.

**Re-confirmed while looking for holes (for the record):** B1/B2/B4/B5 exactly as
doc 01 states; **B3 is narrower than doc 01 implies** — the mismatch is
notebook-only, `final_holdout_eval.py:92-95` already pairs the best checkpoint with
its own vecnorm; commission IS plumbed (`train_ppo.py:112`); baselines share the
same env/costs as the RL agent (`final_holdout_eval.py:117`) so the yardstick is
apples-to-apples; `_SyncVecNormCallback` ordering is correct.

---

## 1. Missed defects & design gaps (N-items)

| # | Item | Type | Phase (doc 04) | Verdict |
|---|---|---|---|---|
| N1 | Gap-through-SL fills | simulator realism | **A/B boundary — decide before baseline** | IMPROVE |
| N2 | Zero-obs truncation bootstrap | training correctness | C (quick win, attributable) | IMPROVE |
| N3 | Gate thresholds don't scale with fold count | ruler | **A** | IMPROVE |
| N4 | Data-end `terminated` vs `truncated` | training correctness | C (bundle with N2) | IMPROVE (minor) |
| N5 | Flat-state action-head noise | learning efficiency | — | KEEP AS-IS |
| N6 | Volume features absent | representation | C (gated on N6-check in Phase 0 data work) | CANDIDATE |
| N7 | No portfolio-level risk overlay | deployment risk | D | IMPROVE (plan gap) |
| N8 | `periods_per_year` hardcodes gold's calendar | metrics | D (fold into doc-02 config) | IMPROVE (trivial) |

### N1 — Gap-through-SL fills `[realism, HIGH]`
- **What:** `_simulate_m1_until_next_decision` fills SL at the exact bracket price
  whenever `low <= sl` (`env_bracket.py:250-256`). If the M1 bar **opens** beyond the
  SL (weekend reopen, news), the real fill is at/near the open — worse than SL. The
  env cannot even see this: it pre-extracts only M1 High/Low, **no Open**
  (`env_bracket.py:83-85`).
- **Measured (census on the on-disk 3.5y LEAN CSV, pure pandas):** 183
  weekend/holiday reopens; median reopen gap 0.33×ATR(H1), 90th pct **1.35×ATR**,
  max **4.3×ATR**; **44 gaps > 1.0×ATR (~monthly)**, 12 > 2.0×ATR. So the tightest
  SL (1.0×ATR) is jumped entirely ~monthly and even the widest (2.0×ATR) several
  times a year — each time the sim under-books the loss by the gap excess. The error
  is **asymmetric** (only ever flatters the strategy).
- **Fix (small):** pre-extract `self._m1_open`; on each M1 bar first check the open:
  long SL fills at `min(open, sl)`, short at `max(open, sl)`; TP stays filled at the
  TP price even when the open is better (keeps the house pessimism, mirrors SL-first).
  Pass the gapped price into `_close_position` (slippage/spread logic unchanged).
- **Future-state fit:** unchanged by doc-02 multi-instrument (per-instrument M1
  arrays gain one more column); strengthens doc-03 §3.7a cost randomization (gap risk
  is the tail the randomization can't fake).
- **Sequencing decision (needs sign-off):** this changes fills → rewards → training
  AND every backtest number. Recommendation: land it in **Phase A, before the
  baseline** — a baseline under known-flattering fills is a false anchor. The
  alternative (Phase C, attributed A/B) is defensible but permanently taints the
  anchor. *Validate:* unit test with a synthetic gap bar; re-run the census
  event-count vs the trade log (every gap-through event in the log must show the
  gapped fill).
- **Cost:** S. **Risk:** low. **Multi-instrument:** helps (FX weekend gaps too).

### N2 — Zero-observation value bootstrap at truncation `[training, MEDIUM]`
- **What:** the env returns an **all-zeros observation** when an episode ends
  (`env_bracket.py:315`). Verified in the installed SB3 2.7.1: `DummyVecEnv` stores
  that zeros vector as `terminal_observation` (`dummy_vec_env.py:70`), `VecNormalize`
  normalizes it (`vec_normalize.py:203-204`), and PPO bootstraps
  `reward += γ·V(terminal_obs)` on truncation (`on_policy_algorithm.py:240-245`).
  Net effect: the GAE target at the **last step of every 2048-step episode** is
  anchored to V(zeros) — a meaningless state — instead of the real next state.
- **Fix (one line):** always return `self._observation()`; index-safe at both ends
  (at termination `i = len-2` is a valid row; at truncation `i < len-2`).
- **Future-state fit:** doc-03 §3.6a warm-start and §3.1b frame-stack both inherit
  the fix; frame-stacking would otherwise *widen* the poisoned vector.
- *Validate:* multi-seed A/B (Phase C, cheap to ride along any run); expect neutral-
  to-small OOS delta with cleaner value loss. **Cost:** S. **Risk:** low.

### N3 — The gate doesn't scale with fold count `[ruler, MEDIUM-HIGH]`
- **What:** `min_consistent_folds=4` and `gate_worst_fold_min_pf=0.9`
  (`config.py:121-124`) were designed for the 5-fold block scheme. The **default**
  sliding path produces ~34 test folds (`train_ppo.py:1079-1087` gates on them):
  breadth "≥4 good folds of 34" is trivially weak while "no single fold PF < 0.9"
  across 34 six-month windows is draconian — one bad window vetoes an otherwise
  robust run. The gate's meaning silently changes with the validation scheme.
- **Fix:** make breadth fractional and the floor a quantile — e.g.
  `min_consistent_fold_frac` (≥ ~60% of folds) and a 10th-percentile-PF floor
  (or "floor with k allowed exceptions"). Keep the block scheme working by deriving
  the old absolute numbers from the fraction.
- **Discipline (doc 03 §7):** pin the new thresholds in Phase A **before** seeing
  baseline results — a gate tuned to pass the baseline is no gate.
- **Future-state fit:** doc-02 gates per-instrument AND aggregate — fraction/quantile
  form composes; absolute counts would break again.
- *Validate:* recompute the gate decision on any multi-fold summary CSV under both
  forms; behavior must be sensible at n=5 and n=34. **Cost:** S. **Risk:** low.

### N4 — Data-end treated as `terminated` `[training, LOW]`
End-of-data sets `terminated=True` (`env_bracket.py:313`), telling the critic the
state has **zero future value** — but running out of CSV is not an absorbing market
state; it should be `truncated` (bootstrapped) like the step-limit path. Rare under
`randomize_start` (most episodes end by step-limit), so impact is small. Bundle with
N2 (same code region, one attribution run).

### N5 — Flat-state action-head noise — **KEEP AS-IS**
When flat with `direction=0` (or holding), the SL/TP heads are still sampled and
entropy-regularized but ignored by the env (brackets bind only at entry,
`env_bracket.py:269-280`). That injects some gradient/credit noise. Fixing it needs
action masking (custom distribution / MaskablePPO-style machinery) — poor
cost/benefit for a 3×3×4 space, and PPO averages the noise out. Documented so it's a
*considered* non-change; revisit only if §3.3a aux-loss work rebuilds the policy
class anyway.

### N6 — Volume/liquidity features absent — **CANDIDATE, gated**
None of the 25 features uses Volume (`features.py:131-140`; the column is loaded
and dropped). Tick volume carries session/participation information, but it is
**vendor-inconsistent** (Dukascopy tick counts ≠ OANDA tick counts ≠ real volume).
Gate: extend doc-04 §1.3's splice-check to compare volume comparability across the
Dukascopy/OANDA overlap; if incomparable, record a one-line REJECT (multi-instrument
breaks otherwise); if comparable, volume-derived ratios (e.g. volume vs its own
EWMA) join a Phase-C feature batch (§3 discipline below).

### N7 — Portfolio-level risk overlay (doc-02 plan gap) `[deployment, MEDIUM]`
Doc 02 deploys "a portfolio of independent books" (§5, §7) with **no aggregate risk
layer**. Six USD-quoted books at 0.5% risk each can stack into one 3% USD bet — the
same correlation doc 02 §3 warns inflates the data count also stacks live exposure.
Add to the doc-02 deployment design: aggregate max-drawdown kill-switch, per-currency
net-exposure cap, max concurrent open books, and (optionally) correlation-aware
risk scaling. Plugs into the ensemble/portfolio layer (`ensemble_analysis.ipynb`
today; a small portfolio module once doc-02 Phase 2 lands). Phase D; measured by the
doc-02 go/no-go portfolio yardstick.

### N8 — `periods_per_year` hardcodes gold's 23h calendar `[metrics, TRIVIAL]`
`_BARS_PER_YEAR` (`config.py:20-31`) bakes in XAUUSD's 23×261 trading calendar; FX
majors trade ~24h — annualized Sharpe/Calmar become slightly incomparable across
instruments. Fold per-instrument `bars_per_year` into the doc-02 §8 instrument-spec
config change. Largely mooted if doc-03 §3.9b's trade-based Sharpe becomes the
selection metric; still fix for reporting hygiene.

---

## 2. Perception & learning-flow audit

**The question (user-posed, correct to ask):** the agent's "understanding of what
price is doing" can only come from (a) what the observation lets it perceive and
(b) what the reward lets it associate. Is a single H1 snapshot with ATR-normalized
features enough?

### 2.1 What the agent perceives today (verified inventory)

One H1 bar → 31 numbers (`env_bracket.py:90-96,148-154`, `features.py:131-140`):

| Perceptual channel | Features | What it can represent |
|---|---|---|
| Trend displacement | `close_ema20/50/200_atr`, `ema20_ema50_atr` | how far price sits from 3 moving anchors |
| Momentum | `macd_hist_atr`, `roc20_atr` | direction/strength of recent push (≤20 bars) |
| Volatility regime | `atr_close`, `atr_fast_slow`, `bb_width_close` | quiet vs stormy, expanding vs contracting |
| Last candle's shape | `range_atr`, wick ratios | rejection/indecision at the current bar |
| Recent path | `ret1..ret5_atr` | the exact last 5 hours, coarser beyond |
| Clock | `tod/dow` sin/cos, 4 session flags | time-of-day/week seasonality |
| Own position | direction, unrealized R, bars-in-trade, dist-to-TP/SL, planned R | trade-management state |

This is a coherent "displacement + momentum + regime + clock" view — genuinely
solid, and its dimensionless discipline is what makes doc-02 pooling possible.

### 2.2 What the agent is blind to (the four real gaps)

1. **Structural position — where price sits in its recent RANGE.** Nothing encodes
   distance to the N-bar high/low or position within the recent range (no
   Donchian/range-position feature exists — verified). The agent can know "1.8 ATR
   above EMA50" but not "2 ticks below last week's high." For a **bracket-placing**
   agent this is the single most decision-relevant blind spot: a TP just beyond a
   prior extreme and a TP in open air have different hit probabilities. → P1.
2. **Intra-bar path — what price did *inside* the hour.** Features are built purely
   from H1 OHLC (`train_ppo.py:60-66`); the M1 stream that the env already loads for
   fills is never summarized into the observation. A grinding 60-minute drift and a
   single 5-minute spike produce the same H1 candle. → P3.
3. **Its own trading costs.** Spread/slippage live only in the fill engine; the
   observation carries no cost signal. Today cost/ATR is *partially* inferable from
   `atr_close` (cost is constant), but the moment doc-03 §3.7a randomizes cost or
   doc-02 varies it per instrument, cost becomes an independent variable the agent
   **must observe to condition on** — otherwise randomization just teaches a blurred
   average. → P2 (a prerequisite interaction the plan missed, not an optional extra).
4. **Market-closure proximity / gap risk.** The N1 census shows ~monthly >1-ATR
   reopen gaps, yet "the market closes in 2 bars" is only implicitly decodable from
   the dow/tod trig corner — hard for a small MLP. If fills start pricing gap risk
   (N1), the agent should be able to *see it coming*. → P4.

Longer history and higher timeframes are already planned (doc 03 §3.1a/b) — not
re-opened here.

### 2.3 How "understanding" actually flows (and its bottleneck)

The agent never predicts price explicitly; comprehension exists only where a reward
gradient rewards it. That channel is **sparse**: non-zero mostly at trade close,
plus the small MTM term. Doc 03 already targets this (aux self-supervised head
§3.3a, delta shaping §3.4a); the audit adds two sharpenings:
- **Multi-horizon aux targets** (extend §3.3a): predict next-bar sign AND ~4-bar
  and ~24-bar direction/realized-range. Trades resolve over hours-to-days, so a
  1-bar-only target under-serves exactly the horizon the brackets live on. Same
  machinery, marginal cost.
- **Adaptivity is a *cadence*, not only a feature set:** the system adapts by
  retraining every `sliding_step_months` and by regime features. Once doc-03 §3.6a
  warm-start makes retraining cheap, the retrain cadence itself becomes tunable —
  measure whether faster adaptation beats 6m (→ P5) instead of assuming it.

### 2.4 New representation candidates (P-items)

All are causal, ATR-normalized/dimensionless (multi-instrument-safe), low-capacity,
and enter ONLY via the batch discipline in §2.5.

**P1 — Range-position / structural-level features. IMPROVE (quick win).**
Donchian-style: `(close − lowN)/(highN − lowN)` and distance-to-N-bar-high/low in
ATR, for one or two N (e.g. 20, 100). Plugs into `features.py:55-143` beside the
EMA distances. *Risk:* moderate collinearity with `close_ema*_atr` — the §2.5
screen decides. *Validate:* screen → single-batch A/B on dev-OOS. **Cost:** S.

**P2 — Observable cost feature (`spread_atr`). IMPROVE (prerequisite-linked).**
Expose current round-trip cost in ATR units in the observation. Implement **with**
doc-03 §3.7a cost randomization and doc-02 per-instrument costs (where cost varies
independently); marginal alone today. Env passes cost into the obs (or a feature
column); must **bypass or join the VecNormalize-mask work** (see P7). **Cost:** S.

**P3 — Intra-bar (M1) microstructure summaries. IMPROVE (medium).**
Per completed H1 bar, from the M1 the pipeline already loads: intra-hour realized
vol vs ATR, path efficiency (net move ÷ sum |M1 moves|), max intra-hour excursion.
Causal (uses only the closed hour). Touches `_load_decision_features`
(`train_ppo.py:46-67`) to aggregate M1→H1 before `prepare_feature_frame`. This is
the audit's strongest "what is price *doing*" addition: it distinguishes grind from
spike. *Risk:* new plumbing; keep to 2–3 features. **Cost:** M.

**P4 — Gap/closure awareness. IMPROVE (pairs with N1).**
`gap_atr` (this bar's open vs prior close, ≈0 intraday, large at reopens — the
census distribution above) and `bars_to_market_close`. Cheap; lets the agent price
the weekend-hold risk N1 starts charging it for. Land in the same feature batch as
P1. **Cost:** S.

**P5 — Adaptation-cadence experiment. IMPROVE (config-only, post-warm-start).**
Sweep `sliding_step_months` (6→3→1) and val-window length once warm-start (§3.6a)
makes it affordable; measure OOS vs cadence. Directly answers "more adaptive to the
changing market" with data instead of intuition. **Cost:** S (compute-gated).

**P6 — Multi-horizon auxiliary targets. IMPROVE (folded into doc-03 §3.3a).**
Not a separate build — it sharpens the already-planned aux-loss (§2.3 above):
predict ~1/4/24-bar direction + realized range instead of next-bar-only, so the
auxiliary gradient covers the horizon the brackets actually live on. Ships with
§3.3a whenever that research bet runs; no independent slot needed.

**P7 — Normalization mask hygiene. IMPROVE (fold into doc-02 tag work).**
`VecNormalize` running-stats all 31 dims — including binary session flags and
position-state dims whose stats are dominated by flat periods; encodings drift as
stats update. Benign today, but doc-02's instrument tag **already requires** a
bypass mechanism (doc 02 §8) — build it as a general "norm mask" and route flags +
position features + P2's cost feature through it. One mechanism, three users.

**P8 — Cross-asset context (e.g., DXY for gold). RESEARCH, deferred.**
The strongest known exogenous driver, but adds a data dependency and per-feature
point-in-time discipline; and doc-02 pooling already injects cross-market structure
implicitly (episodes stay single-instrument, so pairs are never *simultaneously*
observable — a true cross-asset feature would need its own aligned stream). Revisit
after doc-02 Phase 2.

**P9 — Economic-calendar awareness (NFP/FOMC proximity). RESEARCH, gated on data.**
Gold's largest spikes are scheduled events; "minutes-to-known-event" is causal and
cheap *if* a point-in-time calendar source is added to doc-04 §1. Without that
source discipline it's a leakage trap — hence research-tier, data-gated.

### 2.5 Feature-batch discipline (how any of these actually enters)

Observation width **is** capacity — the 2026-06-02 collinearity audit exists
because extra features helped the agent memorize. So:
1. **Screen locally first** (pure pandas, runs today): collinearity vs the existing
   25 (reject |r| ≥ ~0.9, per house precedent) + a cheap predictive sanity check.
2. **One small batch at a time** (≤ ~4 features), A/B'd multi-seed on dev-OOS via
   the doc-04 harness; ship/kill on the pre-committed threshold.
3. **Suggested batch order:** P1+P4 (structure + gap, both S) → P3 (microstructure)
   → P2 (with §3.7a/doc-02 cost work) → N6 volume (if the vendor gate passes).
4. Every batch obeys doc-03 §7's multiple-testing guard (dev surface only; the
   lockbox stays sealed for the final system).

---

## 3. What §2 deliberately does NOT change

The audit **affirms** the core design: dimensionless single-bar features + small MLP
+ sparse-but-honest reward is the right *starting* skeleton, and doc-03's KEEP
verdicts (PPO, small net, fixed sizing, discrete brackets) survive this pass
unchanged. Nothing here adds capacity for its own sake; every P-item is a targeted
answer to a named blind spot, individually killable on OOS evidence. The perception
gaps are the system being *under-informed*, not under-parameterized — the fix is
better inputs, not a bigger brain.

---

## 4. Merge into the execution plan (doc 04)

| Doc-04 phase | Additions from this doc |
|---|---|
| §1 Data | N6 volume-comparability added to the splice-check; P9 calendar source = open decision |
| Phase A (ruler) | **N3** gate re-form (fraction/quantile, pinned pre-baseline); **N1** fill-model decision — recommended to land here, before the baseline |
| Phase B (baseline) | baseline runs under the honest fill model (if N1 lands in A) |
| Phase C (enhancements) | N2+N4 bundle (one attribution run); feature batches P1+P4 → P3 → P2 (§2.5 order); P5 cadence sweep post-warm-start; N6 if gate passes |
| Phase D (multi-instrument) | **N7** portfolio risk overlay in the deployment design; N8 per-instrument bars/year; P7 norm-mask built with the instrument tag; P8 revisited |

Doc-03 §6's tiering is unchanged; where a P-item competes for the same slot as a
doc-03 quick win, doc-03 wins (it's already sequenced) and the P-item joins the
nearest feature batch.

## 5. Open decisions (added to doc 04 §5's list)

- **N1 sequencing:** honest fills before the baseline (recommended) or as an
  attributed Phase-C A/B?
- **Gate form (N3):** fraction + quantile parameters — pin them in Phase A.
- **P9:** source a point-in-time economic calendar in the doc-04 data work, or drop?

## 6. Next action

Unchanged: **Task 2 (B1–B3) proceeds as planned.** These items do not preempt the
ruler work — N3 joins it, N1 needs a sequencing decision at the Phase-A/B boundary,
and everything else queues into C/D where the doc-04 harness can attribute it.
