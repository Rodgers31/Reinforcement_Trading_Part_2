# Multi-instrument training plan

**Goal:** extend the single-instrument (XAUUSD) bot to trade multiple instruments
— FX majors, and optionally metals / an index — and decide *how* it should learn
across them.

**The question this answers:** should we train **one model per instrument**
(separate "lanes"), **one mixed model** across all instruments, or something in
between? And do instruments behave too differently to mix?

All code references are repo-root-relative (e.g. `env_bracket.py:297`). Current-
state facts are verified against the source (see
[`01-baseline-review-and-known-issues.md`](01-baseline-review-and-known-issues.md)).

---

## 1. TL;DR recommendation

It is not a binary choice. Four points on a spectrum:

1. **Specialists** — one model per instrument.
2. **Pooled generalist** — one model, all instruments mixed.
3. **Generalist + instrument tag** — one model, mixed, *conditioned* on which
   instrument it's trading. **(recommended)**
4. **Pretrain-then-fine-tune** — pool to pretrain, then fine-tune a copy per
   instrument.

**Recommended path:** train a **pooled generalist with an instrument tag**,
**benchmark it against per-pair specialists**, optionally **fine-tune per pair**,
and **deploy whichever wins as a portfolio of independent books**. Pooling is
favored because this system's #1 weakness is overfitting a thin signal on limited
data, and pooling N instruments is the strongest available regularizer — while the
instrument tag preserves the ability to behave differently per pair.

```
   Instruments (gold, EURUSD, GBPUSD, USDJPY, …)
            │  each with its OWN cost model
            ▼
   Per-instrument cost  +  ATR-normalized features   ← already dimensionless
            ▼
   Calendar-aligned walk-forward + embargo across ALL pairs
            ▼
   ┌────────────────────────────┬─────────────────────────────┐
   │ Specialists (one per pair) │ Generalist (pooled + tag)    │  ← recommended
   │ = benchmark yardstick      │   + optional per-pair fine-tune
   └────────────────────────────┴─────────────────────────────┘
            ▼                                 ▼
   Portfolio deploy — each model its own book, blend equity
   (low-correlation pairs = the robust risk-adjusted win)
```

---

## 2. Why mixing is technically viable *here* (it usually isn't)

Most bots can't pool instruments: a $2000 gold move and a 0.0001 EURUSD move
aren't comparable numbers, and a cash-PnL reward lets the high-vol instrument
dominate every gradient. **This system already solved both**, by design:

| Property | Verified in code | Why it enables pooling |
|---|---|---|
| Features are dimensionless | all 25 are `÷ATR` or ratios (`features.py:131`) | EURUSD and gold bars land in the same numeric range |
| Reward is in R-units | `(Δequity)/risk_budget` (`env_bracket.py:297`) | a 1R win contributes equally regardless of instrument |
| Sizing is fractional risk | `units = risk_cash/sl_dist` (`env_bracket.py:173`) | auto-adapts to each instrument's price/vol scale |
| Brackets are vol-relative | SL in ATR, TP in R | no absolute pips/dollars baked in |

So the observation, action, and reward are **already a common normalized
language**. The agent never sees raw price. This is exactly the volatility-scaling
/ asset-invariant-feature discipline that makes cross-asset deep-learning trading
models work — and it's why keeping everything in separate lanes would throw away
the biggest advantage the codebase already has.

---

## 3. Why mixing still isn't free

The instinct that "instruments behave differently" is half-right:

- **Normalization removes scale, not behavior.** EURUSD is more mean-reverting
  intraday; gold trends and spikes with fat tails; JPY pairs carry session/carry
  dynamics. A pure generalist *averages* these and can dilute an instrument-
  specific edge. → fix with the **instrument tag** and/or **fine-tuning**.
- **Correlated instruments inflate the data count.** Six USD pairs are not six
  independent datasets — a risk-off regime hits all of them at once. "6× more
  data" is really ~2–3× more *independent* information. Real, but don't oversell.
- **Cost is the hard blocker.** The env applies spread as an absolute price offset
  (`env_bracket.py:157`) with one global `spread_price=0.20` (`config.py:143`).
  Gold's 0.20 on EURUSD-at-1.08 models an ~18% round-trip cost — nonsense.
  **Per-instrument cost is a prerequisite**, ideally expressed in ATR/pip terms so
  it stays comparable across the universe (mirroring the features).

**Resolution:** don't choose "mix vs. separate." **Pool to learn the shared
structure; condition (tag) or fine-tune to recover the specifics.**

---

## 4. How it would learn (mental model)

1. Assemble N instruments, each with its own M1→H1 frames **and its own costs**.
2. Reuse the existing `randomize_start` mechanism (`env_bracket.py:101`), extended
   so **each episode randomly picks one instrument and a window within it**. An
   episode stays on one instrument — positions are path-dependent, so you can't
   carry a gold position into a EURUSD bar.
3. PPO's batch **mixes experience across instruments**. The shared network is
   pressured to find a policy that works on *all* of them → it learns the common,
   instrument-invariant structure. The **instrument tag** (small ID embedding)
   lets that one network shift behavior per pair where pairs genuinely differ.
4. Result: **one model**. Point it at any instrument; it conditions on the tag and
   trades it. Optionally fine-tune a copy per instrument for the last mile.

This is the "foundation model" pattern for trading: one conditioned generalist,
not N blind specialists.

---

## 5. The four architectures

| Architecture | Pros | Cons | Use it as |
|---|---|---|---|
| **Specialists** (one per pair) | captures instrument-specific dynamics; simple | less data each → worse overfitting (the existing weakness); N× training; no transfer | **the benchmark** |
| **Pooled generalist** | N× data → strong regularizer; one model | averages heterogeneous behavior; correlation inflates apparent data | starting point |
| **Generalist + tag** | shared structure **and** per-pair conditioning | slightly larger obs; needs retrain | **primary** |
| **Pretrain → fine-tune** | shared prior + specialization; helps data-poor pairs | more moving parts; per-pair artifacts | last-mile lift |
| *(deployment)* **Portfolio** | diversification = robust Sharpe gain even if edge is a wash | needs per-book bookkeeping | **always, for deploy** |

---

## 6. Phased plan

### Phase 0 — Behavioral-similarity diagnostic (½ day, runs locally today)
Before training anything, measure how differently the instruments behave so
grouping is data-driven, not guessed. Pure pandas/numpy (runs in this env):
- per-instrument **Hurst exponent / variance-ratio** (trending vs mean-reverting),
- **ATR%-of-price** (typical vol),
- autocorrelation of `ret1_atr`,
- cross-instrument **correlation matrix**.

**Output:** pool everything, or pool within **asset-class clusters** (FX majors
together, metals together)? Cluster by behavior, not by asset label.

### Phase 1 — Cheapest high-value win: a 2nd specialist + a 2-book portfolio (1–2 days)
Add per-instrument cost to config, plumb a second CSV (e.g. EURUSD) through the
**existing** single-instrument pipeline, and train it as a separate specialist —
**zero RL-architecture change**. Then blend gold + EURUSD using the ensemble
notebook that **already exists** (`ensemble_analysis.ipynb` already blends fold-
books into a portfolio; instruments slot into the same code). Delivers
diversification — the most robust, model-agnostic improvement — before committing
to the harder build.

### Phase 2 — Pooled generalist (the real build, ~1 week)
- Multi-instrument env sampling (Phase-1 cost plumbing + instrument tag),
- calendar-aligned walk-forward (call `make_sliding_folds` per instrument with
  identical windows; pool the train windows; evaluate per-instrument on each test
  window),
- gate per-instrument **and** in aggregate.

### Phase 3 — Specialize + the decision experiment (~3 days)
Pretrain the generalist, fine-tune a copy per instrument, and run the go/no-go
A/B (see §7).

---

## 7. The go/no-go decision experiment

This is the experiment that answers "is mixing worth it?" **with data**, not
opinion. On the **same out-of-sample windows**, compare:

| Variant | What it tests |
|---|---|
| Mean-of-specialists | the baseline "separate lanes" |
| Pooled generalist | does raw data-pooling help? |
| Generalist + tag | does conditioning recover per-pair behavior? |
| Fine-tuned generalist | does specialization add a last-mile lift? |

**Decision rule:** adopt the generalist only if it **beats the specialists' mean
OOS**. Regardless of the winner, **deploy as a portfolio** — the diversification
of low-correlation books improves risk-adjusted return even when the modeling is a
wash. This mirrors how the repo already uses baselines as yardsticks: if the
fancier thing can't beat the simple thing OOS, it added nothing.

---

## 8. Concrete code changes

| File | Change | Effort |
|---|---|---|
| `config.py` | `csv_path` (single) → a list of instrument specs, each with its own `spread`/`slippage`/`commission` (ideally in ATR/pip terms). `config.py:39`, `config.py:143-145` | small |
| `data_loader.py` | load/resample N CSVs keyed by instrument; MT4 parsing/tz logic reused as-is | small |
| `features.py` | **reusable almost unchanged** (the big win). Add an instrument-ID feature; ideally have the ID **bypass VecNormalize** (or be a learned embedding) so a one-hot isn't distorted by running-mean normalization | small |
| `env_bracket.py` | biggest change: hold `{instrument: (decision_df, m1_df, cost)}`; `reset()` picks instrument + window; per-instrument cost flows into `_entry_price`/`_exit_price` (`env_bracket.py:156-161`) | medium |
| `train_ppo.py` | fold logic slices the **same calendar window** across all instruments and builds a pooled training env; VecNormalize/PPO/callbacks/gate largely reuse | medium |
| validation | enforce calendar-aligned embargo across the **whole universe** (sliding WF is already calendar-based — the right foundation) | small |
| notebooks | extend `ensemble_analysis.ipynb` (already blends books) to blend per-instrument books | small |

---

## 9. Prerequisites & gotchas

1. **Per-instrument cost is non-negotiable** — get it wrong and the agent
   optimizes against fake friction. Express it in ATR/pip terms.
2. **Calendar-aligned splits across the universe** — or correlated instruments
   leak regimes train↔test.
3. **Don't trust the naive data-multiplier** — correlation means effective sample
   gain < N×.
4. **Adding the instrument tag changes obs shape** (currently 31-dim) → forces a
   full retrain. The diagnostics already warn that obs-shape changes require retrain.
5. **Episodes must stay single-instrument** — the env is path-dependent; mixing
   instruments mid-episode would corrupt open positions.
6. **The portfolio blend is the robust win** — even if "mix vs. separate" ties on
   edge, uncorrelated books improve Sharpe reliably.
7. **Fix B1–B3 first (doc 01).** Multi-instrument multiplies the number of
   models/gates/vecnorm pairings; shipping the current gate/vecnorm bugs into an
   N-instrument pipeline multiplies the blast radius.

---

## 10. Open questions (need product input)

- **Universe:** majors only, or majors + metals + an index? (changes whether
  clustering is worth the complexity.)
- **Count:** how many instruments initially? (drives Phase-0 clustering value.)
- **Data parity:** gold has ~23y; some pairs have shorter/cleaner history —
  affects calendar alignment (early folds have fewer instruments) and makes the
  pretrain→fine-tune path more valuable for data-poor pairs.
- **Broker cost realism:** do we have real per-instrument spread/commission, or do
  we model conservative constants?

---

## 11. Next action

Recommended: **implement Phase 0** (the behavioral-similarity diagnostic) as a
standalone pandas script — it runs in the current environment today and turns the
"pool vs. cluster" decision into a data-driven one before any RL infra changes.
