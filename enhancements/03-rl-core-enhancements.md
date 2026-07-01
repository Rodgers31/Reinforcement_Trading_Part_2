# RL core enhancements — a smarter, sample- & compute-efficient agent

**Subject:** making the RL system *itself* more capable, more sample-efficient,
cheaper to train, and better at generalizing — independent of, but compatible
with, the multi-instrument direction in
[`02-multi-instrument-training-plan.md`](02-multi-instrument-training-plan.md).

**This is a planning artifact only.** No code was written, edited, or run; no
training was started. Every current-state claim is traced to source (`file:line`),
not the README/docstrings (several are stale — see
[`01-baseline-review-and-known-issues.md`](01-baseline-review-and-known-issues.md)).

**Environment limit:** this machine can't run the RL stack (no
gymnasium/sb3/torch), only the pandas/numpy core. Where a claim needs training to
confirm, it is flagged and the confirming experiment is specified.

**Stance:** this is an audit, not a rewrite mandate. Doc 01 rates the data
handling, causal-feature discipline, embargoed walk-forward, and consistency-based
checkpointing as above average. Several components are marked **KEEP AS-IS** on
purpose — a forced change to an already-good part is a failure mode, not
thoroughness. Everything is judged on **out-of-sample generalization**, never
train fit; anything whose main effect is more capacity to memorize the thin signal
is flagged or rejected.

---

## 1. Current state (verified) — the baseline every proposal changes FROM

| Component | Current behavior | Source |
|---|---|---|
| **Observation** | Single H1-bar snapshot: 25 market features + 6 position-state features = **31-dim Box**. No multi-bar sequence, no higher-timeframe context. `ret1..5_atr` encode 5 bars of return only. | `env_bracket.py:90-96`, `148-154`, `features.py:131-140` |
| **Action** | `MultiDiscrete([3,3,4])` = (dir ∈ flat/long/short, SL ∈ {1.0,1.5,2.0}×ATR, TP ∈ {1,1.5,2,3}×R). Bracket set at entry only; no trailing/partial/scale. | `env_bracket.py:87`, `config.py:137-138` |
| **Sizing** | Fixed fractional: `units = (equity·0.5%) / sl_distance`. Not an action. | `env_bracket.py:174`, `config.py:142` |
| **Reward** | `(Δequity)/risk_unit + 0.01·unrealized_R − 0.00002` per step. Sparse realized term (only on close) + dense MTM shaping (level, not delta) − flat holding penalty. | `env_bracket.py:297-300`, `config.py:148-149` |
| **Policy/value net** | SB3 `MlpPolicy`, hidden layers **(128, 64)**, `weight_decay=1e-5`. Feedforward; **no recurrence/attention/frame-stack**. | `train_ppo.py:437,455-456`, `config.py:164` |
| **Algorithm** | PPO, generalization-first preset: `lr=6e-5` linear-decay, `ent=0.03`, `n_epochs=5`, `clip=0.1`, `target_kl=0.025`, `gamma=0.99`, `gae=0.95`, `n_steps=2048`, `vf_coef=0.5`. Obs+reward `VecNormalize`. | `train_ppo.py:436-458,372`, `config.py:157-164` |
| **Checkpoint selection** | `_ConsistencyEvalCallback`: keep the checkpoint maximizing `min(q_train, q_val)`, `q = reward − dd_penalty·maxDD%`, only when both legs profitable and ≥`min_trades=5`. | `train_ppo.py:236-246,142-289` |
| **Validation** | Default = sliding walk-forward: train 5y → val 6m → test 6m, slide 6m, ~34 folds, each trained **from scratch**; 200-bar embargo; deploy gate on test folds. | `train_ppo.py:929-1099`, `config.py:110-113,121-124` |
| **Cost model** | Fixed constants: `spread=0.20`, `slippage=0.02`/side, in absolute price units; spread re-paid at exit; SL-first when both hit a candle. | `env_bracket.py:156-161,250-256`, `config.py:143-144` |
| **Seeds** | Single seed (`42`) everywhere; no multi-seed averaging or PPO hyperparameter search. | `train_ppo.py:311,441` |

Two facts anchor most of what follows:
- **The learning signal is sparse.** The realized reward term is non-zero only on
  a trade close; between trades the agent gets only the small MTM shaping. Sample
  efficiency is therefore genuinely constrained.
- **`gamma=0.99` at H1** ⇒ effective horizon ≈ 100 bars ≈ ~4 trading days — long
  relative to typical trade durations, but not obviously wrong.

---

## 2. Vision / north star

**What "smarter, more efficient, more advanced" means for THIS system** — not a
bigger network (capacity is the enemy of a thin signal), but:

**North-star capabilities**
1. **Context-aware** — the policy perceives recent *history*, *higher-timeframe*
   trend/vol, and (per doc 02) *instrument identity*, so one policy adapts its
   behavior to conditions instead of reacting to a single bar in isolation.
2. **Sample-efficient** — it extracts more learning per env-step (denser/auxiliary
   signal, better credit assignment), because the reward is sparse.
3. **Compute-efficient** — a full walk-forward costs a fraction of today's ~100M
   env-steps via warm-starting and smarter per-fold budgeting.
4. **Honestly selected & robust** — the shipped model is chosen by an
   OOS-aligned, multi-seed-stable metric, and the reporting layer cannot silently
   ship a gate-failed or mis-normalized model (doc 01 B1–B3).
5. **Multi-instrument-ready** — every representation/architecture choice composes
   with the doc-02 pooled generalist + instrument tag.

**Guiding principles:** generalization-first · cost-realistic · walk-forward-honest
· multi-instrument-ready · **parsimony** (never add capacity to memorize noise).

**Yardsticks (how we know we got there)** — all measured on the *same* sliding
walk-forward **test** windows, against **(a)** the current PPO and **(b)** the
existing EMA/ATR baselines:
- risk-adjusted OOS return (return / max drawdown; profit factor), not raw return;
- **sample efficiency**: OOS quality at a fixed env-step budget (area under the
  learning curve);
- **compute**: env-steps / wall-clock per fold to reach a target OOS;
- **stability**: variance of OOS metrics across seeds and across folds.

A change ships only if it **beats the current model OOS at equal-or-less compute**,
or buys a real efficiency/robustness gain without hurting OOS. This is the same
go/no-go discipline as doc 02 §7.

---

## 3. Enhancement assessment by dimension

Template per item: **Verdict** · what/where · why it helps the vision · expected
impact (directional, no fabricated numbers) · cost (S/M/L) · risk · multi-instrument
fit · how to validate. KEEP items stop after the reason.

### 3.1 Observation / representation

**(a) Higher-timeframe context features — IMPROVE (quick win).**
- Add a handful of causal H4/D1 features (trend distance, vol regime) as extra
  ATR-normalized columns, reusing the existing `features.py` machinery on a coarser
  resample. Plugs in at `features.py:55-143`, consumed unchanged by the obs at
  `env_bracket.py:148-154`.
- *Why:* the agent currently sees only the H1 bar; macro context is the cheapest
  way to make it *condition* rather than react. Low capacity, high signal.
- *Impact:* better regime-conditioning, especially fewer counter-trend entries.
- *Cost:* S. *Risk:* low (a few dimensionless features; keep the count small to
  avoid re-inflating the obs the collinearity audit trimmed).
- *Multi-instrument:* helps (instrument-agnostic).
- *Validate:* add features, retrain, compare OOS vs current. Local pre-check
  (pandas): Spearman of each HTF feature vs forward return, and collinearity vs
  existing columns, before spending training.

**(b) Short multi-bar history (frame-stack) — IMPROVE, small; RESEARCH if pushed far.**
- SB3 `VecFrameStack(k)` stacks the last *k* observations; no architecture change.
  Currently absent (confirmed: no `FrameStack`/`n_stack` anywhere).
- *Why:* gives the MLP the recent trajectory (momentum/acceleration) beyond the
  hand-coded `ret1..5`. Better credit assignment on path-dependent setups.
- *Impact:* modest; helps if intra-window sequence matters.
- *Cost:* S to wire; but obs dim ×k (31→~124 at k=4) **adds capacity → overfitting
  surface**. Keep k small (3–4).
- *Risk:* medium (capacity). Pair with the existing weight-decay/entropy stack.
- *Multi-instrument:* neutral.
- *Validate:* sweep k ∈ {1,3,4} on OOS folds; adopt only if k>1 beats k=1 OOS. If
  history clearly helps but a stacked MLP overfits, that motivates recurrence (§3.2).

**(c) Regime/trend-vs-range feature — KEEP AS-IS (mostly covered).**
- The obs already carries trend (`close_ema20/50/200_atr`), vol regime
  (`atr_fast_slow`), and sessions. An explicit efficiency-ratio/Hurst feature is
  marginal on top of these. Revisit only if §3.1a/b show the agent is
  regime-blind. Not worth a dedicated change now.

### 3.2 Policy / value architecture

**Base MLP (128,64) — KEEP AS-IS.** Deliberately small to regularize the thin
signal (`config.py:164` comment, verified). For the current single-bar obs, adding
width/depth is precisely the "memorize noise" failure mode. Right-sized; leave it.

**Instrument-tag embedding — IMPROVE (owned by doc 02).** A small learned embedding
for instrument ID, concatenated to features, is the one architecture addition
clearly worth it — but it belongs to the multi-instrument build (doc 02 §8). Noted
here for completeness; ideally the ID **bypasses `VecNormalize`** (a one-hot
distorted by running-mean norm is a subtle bug waiting to happen).

**Recurrence / attention over history — RESEARCH LONG-SHOT.**
- `RecurrentPPO` (LSTM) or a small attention encoder over the bar window; replaces
  the feedforward trunk. Currently absent.
- *Why:* parameter-efficient sequence memory — the "right" way to use history if
  §3.1b proves history matters but stacking overfits.
- *Impact:* potentially the biggest representational gain; also the biggest risk.
- *Cost:* L (training instability, ~2–3× compute, careful masking across episode
  resets). *Risk:* high (overfitting + optimization).
- *Multi-instrument:* neutral, but interacts with per-episode instrument sampling
  (state resets at instrument boundaries) — needs care.
- *Validate:* only after §3.1b establishes a history signal; compare RecurrentPPO
  vs best frame-stack MLP on OOS at matched compute. Kill if it doesn't beat the
  MLP OOS.

### 3.3 Learning algorithm & objective

**PPO + generalization-first preset — KEEP AS-IS (algorithm).** PPO is a sound
choice; the preset (tight clip, few epochs, KL brake, LR decay, entropy, weight
decay) is a coherent, above-average generalization stack. Do **not** swap for
SAC/DQN/distributional RL — no clear benefit, more instability, cargo-cult for this
problem.

**(a) Auxiliary self-supervised loss — RESEARCH BET (high potential).**
- Add a small auxiliary head off the shared trunk predicting an instrument-
  normalized self-supervised target (e.g. next-bar sign of `ret1_atr`, or next-bar
  realized range), trained with a small-weight loss alongside PPO. Does **not**
  change the policy/reward objective.
- *Why:* the reward is sparse (§1); an auxiliary task densifies the gradient and
  shapes the representation — a standard sample-efficiency lever (UNREAL-style
  auxiliary tasks).
- *Impact:* better sample efficiency / representation; effect bounded by how much
  short-horizon signal exists (an efficient market limits it — but even weak
  targets shape features).
- *Cost:* M (custom policy/loss in SB3). *Risk:* medium (a mis-weighted auxiliary
  loss can distract the policy).
- *Multi-instrument:* neutral/helps (target is R/ATR-normalized).
- *Validate:* OOS sample-efficiency curve (OOS quality vs env-steps) with vs
  without the aux loss; adopt only if it reaches equal OOS in fewer steps or higher
  OOS at the same budget.

**(b) OOS-honest hyperparameter search — IMPROVE (with caveats).**
- No systematic PPO HPO exists (confirmed; only the baseline grid in
  `baselines.py`). The preset is hand-tuned. A small search over
  `lr, ent_coef, gamma, clip, n_epochs` could find a better generalization point.
- *Why:* `gamma`, entropy, and clip in particular trade off directly against
  overfitting; hand-tuning likely left value on the table.
- *Impact:* modest-to-moderate; mostly a robustness/So win.
- *Cost:* M–L (many training runs). *Risk:* **meta-overfitting the validation** if
  tuned and judged on the same folds.
- *Multi-instrument:* neutral.
- *Validate:* **nested** walk-forward — tune on inner folds, judge on outer
  held-out folds; report the outer-fold OOS only. Cheap only after warm-start (§3.6a).

**(c) Reward/advantage normalization — KEEP AS-IS.** `VecNormalize` already
normalizes obs and reward (`train_ppo.py:372`); GAE + reward-norm are appropriate.

### 3.4 Reward design

**(a) Potential-based (delta) MTM shaping — IMPROVE (clean quick win).**
- Change the shaping term from the *level* of unrealized R (`env_bracket.py:299`)
  to the *change* in unrealized R (potential-based shaping, Ng et al.).
- *Why:* doc 01 R4 — the level form integrates over holding time and biases toward
  holding. The delta form telescopes to realized PnL, is policy-invariant in
  theory, and keeps the dense signal without the holding bias.
- *Impact:* cleaner incentives, less spurious "hold winners" pressure; small but
  principled.
- *Cost:* S. *Risk:* low.
- *Multi-instrument:* neutral.
- *Validate:* OOS trade duration + return vs current; expect similar/better OOS
  with less duration inflation.

**(b) Drawdown / risk-adjusted reward aligned with selection — IMPROVE (high-leverage).**
- The agent maximizes cumulative R, but we *select* on drawdown-penalized
  consistency and *deploy* on risk-adjusted return — a reward/metric misalignment.
  Add an intra-episode drawdown penalty, or move to an online differential
  Sharpe-style reward (Moody & Saffell) so the agent optimizes the thing we
  actually judge.
- *Why:* aligning the training objective with the deployment yardstick is one of
  the most reliable ways to improve *deployed* performance.
- *Impact:* potentially large on risk-adjusted OOS (the metric we care about).
- *Cost:* M. *Risk:* medium — risk-aware rewards can suppress trading toward a
  degenerate do-nothing policy; the existing both-legs-profitable/min-trades gate
  (`train_ppo.py:243-246`) partially guards this, but tune carefully. Also: training
  wraps the env in `VecNormalize(norm_reward=True)` (`train_ppo.py:372`), which
  rescales rewards by a running reward-std — that directly fights a differential-
  Sharpe-style signal (it re-normalizes away the very variance the reward is trying
  to price), so any risk-adjusted reward must disable or adapt reward normalization.
- *Depends on:* honest mark-to-market equity (§3.9a) to make drawdown meaningful.
- *Multi-instrument:* neutral/helps (R-normalized).
- *Validate:* OOS return/maxDD and Sharpe vs current; watch trade count for collapse.

**(c) Turnover/cost awareness — KEEP AS-IS.** The env already pays real spread on
every fill (a "1R" TP nets ~0.94R, doc 01), so churn is already penalized
economically; the flat holding penalty is minor. No dedicated change needed.

### 3.5 Action space & trade management

**Fixed fractional sizing — KEEP AS-IS.** Making size an action is a classic
overfitting *amplifier* — it's where agents most easily learn to over-leverage
in-sample luck and fatten tails. For a thin signal, fixed risk is a feature. If
ever added, do it as a coarse 2–3-level bucket with hard caps, as a late research
item — not now.

**Discrete bracket grid — KEEP AS-IS.** {1,1.5,2}×ATR SL and {1,1.5,2,3}R TP is a
sensible coarse grid; continuous brackets enlarge the action space and overfitting
surface for little gain.

**Trailing / partial exits / bracket adjustment — RESEARCH LONG-SHOT.** Real
trade-management upside, but expands the action space and complicates the M1 fill
sim. Defer until the representation/efficiency work lands; weigh explicitly against
overfitting. *Validate* (if pursued): OOS vs the entry-only bracket at matched
compute.

### 3.6 Sample & compute efficiency (the ~100M-step problem is first-class)

**(a) Warm-start across folds — IMPROVE (headline compute win).**
- Sliding folds overlap heavily (5y windows stepping 6m ⇒ ~90% shared data) yet
  each `train()` builds a **fresh** PPO (`train_ppo.py:996-1007,436`). Initialize
  fold *k+1* from fold *k*'s weights + VecNormalize stats, then train fewer steps.
- *Why:* the single biggest lever on the ~100M-step cost; also **more realistic** —
  live retraining would warm-start from the deployed model, not from scratch.
- *Impact:* potentially large reduction in per-fold steps to reach target OOS.
- *Cost:* M. *Risk:* low-medium — beyond mild fold coupling (mitigated by
  re-selecting the checkpoint per fold; the test window stays untouched, so per-fold
  OOS honesty holds), warm-starting changes *what the walk-forward measures*:
  from-scratch-per-fold tests "can the method rediscover an edge from scratch each
  era," while warm-start makes the ~34 test windows non-independent — shrinking the
  aggregate error bars and letting a stale regime persist across folds. Worth the
  compute win, but log the lost fold-independence as a real tradeoff, not a footnote.
- *Multi-instrument:* helps (the pooled generalist still walks forward in time).
- *Validate:* OOS per fold at reduced step budget vs from-scratch at full budget;
  adopt if OOS holds at materially lower compute.

**(b) Per-fold step budget + early-stop — IMPROVE.**
- 3M steps/fold (`train_ppo.py:930`) may exceed what a warm-started fold needs. Use
  the consistency-eval learning curve to early-stop folds that plateau; tune the
  budget down.
- *Cost:* S. *Risk:* low. *Multi-instrument:* neutral.
- *Validate:* OOS vs step budget; pick the knee of the curve.

**(c) Throughput: more envs + GPU batch — IMPROVE (config-level).**
- Env simulation is the bottleneck and the M1 lookup is already optimized
  (searchsorted + numpy, `env_bracket.py:79-85`). Raising `n_envs` (4→8–16) on a
  many-core box and tuning `batch_size` for the device is straightforward wall-clock.
- *Cost:* S. *Risk:* low (reproducibility across `n_envs` — fix seeds/document).
- *Multi-instrument:* helps (more parallel books/instruments per rollout).
- *Validate:* wall-clock/throughput benchmark; confirm OOS unchanged at fixed steps.

**(d) `sliding_step_months` knob — KEEP AS-IS (already configurable).** Raising it
(6→12) halves fold count (`config.py:113` documents this). A dial, not a change.

### 3.7 Generalization / overfitting defenses (the #1 weakness)

**Existing stack — AFFIRM.** Small net, weight decay, entropy, 5 epochs, tight
clip, KL brake, LR decay, collinearity-trimmed features, 200-bar embargo, and the
consistency gate together form a genuinely above-average regularization stack. Keep
all of it.

**(a) Cost / slippage domain randomization — IMPROVE (top regularizer, cheap).**
- Sample `spread`/`slippage` per episode from a plausible range instead of fixed
  constants (`env_bracket.py:70-72,156-161`).
- *Why:* forces the policy to be robust to execution cost rather than exploiting
  one exact cost number; a classic sim-to-real robustness technique. **Directly
  prepares for multi-instrument**, where per-instrument cost is different and
  uncertain (doc 02 §3).
- *Impact:* better OOS robustness, less brittle bracket selection.
- *Cost:* S. *Risk:* low.
- *Multi-instrument:* strongly helps.
- *Validate:* OOS under held-out (unseen) cost levels vs the fixed-cost model.

**(b) Multi-seed training + ensembling — IMPROVE (high-confidence).**
- Train a few seeds per fold and blend as a portfolio — infra already exists
  (`ensemble_analysis.ipynb` blends books). Currently single seed
  (`train_ppo.py:441`).
- *Why:* seed variance is real in RL; ensembling is one of the most reliable OOS
  improvements and shrinks the run-to-run luck that a single seed bakes in.
- *Impact:* lower OOS variance, usually better median OOS.
- *Cost:* M (N× training — mitigated by warm-start §3.6a). *Risk:* low.
- *Multi-instrument:* composes with the portfolio deployment (doc 02).
- *Validate:* OOS mean/variance of the seed-ensemble vs single seed.

**(c) Feature-noise augmentation — IMPROVE (modest); mirror augmentation REJECTED.**
- Small Gaussian noise on observations is a light regularizer. **Do NOT** use
  long/short mirror augmentation: gold has a secular uptrend, so mirroring
  PnL/direction injects bias.
- *Cost:* S. *Risk:* low (noise) / high (mirror — rejected).
- *Validate:* OOS with vs without noise; small expected effect.

**(d) Pooling across instruments — doc 02.** The biggest *data-based* defense;
owned by doc 02. Referenced, not duplicated.

### 3.8 Evaluation, selection & validation

**Consistency-gate design — AFFIRM.** `min(q_train, q_val)` with a drawdown penalty
and a do-nothing guard (`train_ppo.py:236-246`) is a thoughtful, above-average
selector. Keep the design.

**(a) PREREQUISITE — fix doc 01 B1–B3.** Before any selection/reward change can be
*trusted*: the gate doesn't actually gate (B1), and two eval paths pair a model
with the wrong `VecNormalize` (B2/B3). Any OOS comparison built on a mis-normalized
or wrongly-shipped model is meaningless. These are prerequisites, not enhancements.
- *Validate:* unit-level — assert `gate_passed` is honored downstream; assert the
  best checkpoint is always paired with `best_model_vecnorm.pkl`.

**(b) Multi-seed selection — IMPROVE.** Select on the mean/median across seeds, not
one seed's peak (pairs with §3.7b). Reduces selection-time luck.
- *Cost:* S on top of §3.7b. *Risk:* low. *Validate:* OOS stability of the selected
  model across repeated runs.

**(c) Selection ↔ deployment metric alignment — IMPROVE.** If §3.4b adopts a
risk-adjusted reward, align the gate's `q` with the same risk-adjusted metric so
train objective, selection, and deployment all agree. *Cost:* S. *Validate:*
consistency of ranking between selection score and OOS deployment metric.

### 3.9 Robustness / risk / regime & realism

**(a) Mark-to-market equity for honest drawdown — IMPROVE (unblocks §3.4b).**
- Equity updates only on trade close (`env_bracket.py:302-309`), so max drawdown
  understates true intra-trade drawdown (doc 01 R1). Track an MTM equity curve
  (include open-position unrealized) for the risk metrics used in selection/reward.
- *Why:* makes drawdown/risk metrics honest — a prerequisite for a credible
  drawdown-aware reward (§3.4b) and gate.
- *Impact:* truer risk accounting; may change which checkpoints are selected.
- *Cost:* M. *Risk:* low. *Multi-instrument:* neutral/helps.
- *Validate:* given a saved run (there is none in the repo yet — produce or locate
  one first), reconstruct the MTM equity path in pandas from the **trade log + price
  series** — the saved equity curve is realized-only (`env_bracket.py:302-309`) and
  can't be used alone — then compare its maxDD to the realized maxDD. An H1-close
  reconstruction still misses M1 intra-trade extremes, so the gap it reports is a
  **lower bound** on the true understatement.

**(b) Sharpe definition for selection/reporting — IMPROVE (minor).** The per-bar
Sharpe annualized by full bar count (`evaluate.py:27-34`, doc 01 R2) is noisy on a
mostly-flat series. Prefer a trade-based or properly-scaled Sharpe for selection and
reporting. *Cost:* S. *Validate:* compare rankings; adopt if more stable.

**(c) Intrabar fill realism — KEEP AS-IS (hedge via §3.7a).** The raw-mid detection
+ spread re-application (doc 01 R3) is an approximation; a true bid/ask intrabar
model is heavy. Cost-domain-randomization (§3.7a) hedges the cost uncertainty at a
fraction of the effort. Log as realism debt, don't build now.

**(d) Minor hygiene — IMPROVE (trivial).** Empty-split guard (doc 01 R5) and the
tod-local-vs-session-UTC mismatch (R6) are cheap correctness fixes; batch them with
whatever touches those files.

---

## 4. Deliberately KEEP AS-IS (considered decisions, not omissions)

- **PPO + generalization-first preset** — sound algorithm, coherent anti-overfit
  preset (§3.3).
- **Small MLP (128,64)** for the current single-bar obs — right-sized; more
  capacity = memorization (§3.2).
- **Fixed fractional sizing** — a strength for a thin signal, not a limitation (§3.5).
- **Discrete bracket grid** — coarse is good (§3.5).
- **Consistency-gate design** — above average; fix the *bugs around it*, keep the
  design (§3.8).
- **Causal features + embargo + collinearity trims** — doc 01 rated strong; leave.
- **Cost realism in the reward** (spread on every fill) — already economically
  honest (§3.4c).

## 5. Anti-recommendations (cargo-cult to avoid)

- Swapping PPO for off-policy/distributional RL — instability for no clear gain.
- A deep/wide network or a big transformer on 25 features — memorizes noise.
- Continuous action space or free position sizing — overfitting amplifiers.
- Long/short mirror data augmentation on a trending asset — injects bias.
- HPO tuned and judged on the same folds — meta-overfits the validation.

---

## 6. Prioritization & sequencing

| Tier | Change | Dim | Cost | Confidence |
|---|---|---|---|---|
| **Quick wins** | Fix B1–B3 (prerequisite) | 3.8a | S–M | high |
| | Mark-to-market equity + honest metrics | 3.9a/b | M | high |
| | Potential-based (delta) MTM shaping | 3.4a | S | high |
| | Cost/slippage domain randomization | 3.7a | S | high |
| | Higher-timeframe context features | 3.1a | S | med-high |
| | Throughput: more envs + GPU batch | 3.6c | S | high |
| **High-leverage bets** | Warm-start across folds | 3.6a | M | high |
| | Multi-seed training + ensembling | 3.7b/3.8b | M | high |
| | Risk-adjusted reward ↔ selection alignment | 3.4b/3.8c | M | med-high |
| | Instrument-tag embedding | doc 02 | M | high |
| | OOS-honest (nested) hyperparameter search | 3.3b | M–L | med |
| **Research long-shots** | Auxiliary self-supervised loss | 3.3a | M | med |
| | Sequence memory: frame-stack → recurrence | 3.1b/3.2 | S→L | med→low |
| | Trade management (trailing/partial/sizing) | 3.5 | L | low |

**Recommended ordering (respects dependencies; interleaves with doc-02 phases):**
1. **Fix B1–B3** — until the reporting layer is honest, no OOS comparison can be
   trusted. Foundation for everything.
2. **Mark-to-market equity + honest Sharpe** — unblocks credible risk metrics and
   the risk-aware reward. (Pandas-measurable given a saved run — see §3.9a/§7 for the
   caveats: none exists in-repo yet, and it needs the trade log + prices, not the
   realized-only equity curve.)
3. **Cheap regularizers/hygiene**: delta shaping, cost-domain-randomization, HTF
   features. Each independently OOS-testable.
4. **Compute**: warm-start + throughput — makes every later experiment (multi-seed,
   HPO) affordable.
5. **Multi-seed ensembling** (now cheap post-warm-start) + selection on the
   ensemble.
6. **Risk-adjusted reward** aligned with the (now honest) selection metric.
7. **Interleave doc-02 Phase 0–2**; add the **instrument-tag embedding** when
   pooling begins.
8. **Research bets last**, each gated by its OOS experiment: auxiliary loss, then
   (only if history proves it earns its keep) sequence memory; trade-management
   last of all.

---

## 7. The validation harness (how every change is judged)

One shared protocol so results are comparable:
- **Same sliding walk-forward test windows** as the current pipeline; test data
  stays sealed (doc 01 confirms the geometry is honest).
- **Compare against two references** every time: current PPO and the EMA/ATR
  baselines.
- **Metrics:** risk-adjusted OOS (return/maxDD, PF) first; then sample efficiency
  (OOS vs env-steps) and compute (steps/wall-clock to target); then variance across
  seeds and folds.
- **Ship rule:** beat the current model OOS at equal-or-less compute (or a real
  efficiency/robustness gain with no OOS regression). Otherwise kill it — a
  documented negative result is a valid outcome (doc 02 §7 discipline).
- **Multiple-testing guard:** the ship rule is applied per-item across the ~13–15
  enhancements here, all judged on the same ~34 overlapping sliding-OOS windows —
  so some will "win" OOS by luck (researcher degrees of freedom); this is the
  meta-overfitting §3.3b flags for HPO, generalized to the whole program. Standing
  rule: treat the sliding-OOS as a **validation/dev surface for ranking ideas**,
  pre-commit the ship threshold, and discount marginal single-idea OOS wins.
  Reserve the truly sealed final holdout (`test_frac`) for the **one** final
  combined system, revealed once — never per idea. This is the repo's own
  "seal the test, reveal once" discipline (`final_holdout_eval.py`) applied to the
  enhancement program itself.
- **Locally runnable now (no RL stack):** HTF-feature predictive/collinearity
  screens (§3.1a) run on the raw data today; the MTM-vs-realized drawdown gap
  (§3.9a) is pandas-only too, but needs a saved run first (none exists in the repo
  yet) and must be rebuilt from the trade log + prices, so it yields a lower bound
  on the true gap.

---

## 8. Open questions (need input)

- **Compute envelope:** what wall-clock/$ per full walk-forward is acceptable? Sets
  how hard to push warm-start (§3.6a) and multi-seed (§3.7b).
- **Reward objective:** optimize raw risk-normalized PnL, or explicitly
  risk-adjusted (drawdown/Sharpe-aware, §3.4b)? This is a product decision about
  what "good" means and should match the deployment yardstick.
- **History horizon:** is there reason to believe multi-bar sequence matters at H1
  beyond `ret1..5`? Cheap to probe (§3.1b) before committing to recurrence.
- **Seeds & ensembling:** how many seeds/members can the compute budget support
  (drives §3.7b and the deployment portfolio size)?
- **Sequencing vs doc 02:** land the single-instrument core wins first, or start
  multi-instrument pooling in parallel? (Recommendation: core wins 1–4 first —
  they make the multi-instrument build cheaper and more honest.)

## 9. Next action

Implement the **foundation** before any "smarter-agent" work: **fix B1–B3 and
switch the risk accounting to mark-to-market equity** (doc 01 B1–B3 + §3.9a). Until
the reporting layer is honest and drawdown is measured truthfully, no OOS
comparison — for any enhancement in this doc — can be believed. The zero-training
first step needs a saved run (none in the repo yet): produce or locate one, then
measure the MTM-vs-realized max-drawdown gap in
pandas — reconstructed from the **trade log + prices**, since the saved equity curve
is realized-only (`env_bracket.py:302-309`), and read as a lower bound on the true
understatement. The result directly informs whether §3.4b (risk-adjusted reward) is
worth prioritizing.
