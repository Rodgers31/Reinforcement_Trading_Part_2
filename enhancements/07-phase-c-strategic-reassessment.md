# 07 — Phase-C Strategic Reassessment

**Date:** 2026-07-05 · **Branch:** `execution/phase-a` · **Trigger:** 3 Phase-C A/Bs, all NO SHIP.

## Bottom line

Three reward/cost A/Bs (turnover-flat, cost-randomization, flip-aware turnover) all failed the
ratified ship rule. Together they **empirically confirmed** the system's #1 weakness —
*thin-signal transfer failure* (fits train, fails OOS; train+val− = 33%) — **and** that
reward/cost shaping **cannot fix it** (it only trims cost drag on a near-zero-edge system). The
enhancement docs, written *before* these A/Bs, already predicted exactly this and prescribed the
fix: **more DATA via multi-instrument pooling** — doc 03 §3.7d ("the biggest *data-based* defense")
and doc 02 §1 ("this system's #1 weakness is overfitting a thin signal on limited data, and
pooling N instruments is the strongest available regularizer"). That lever is **untried** and needs
data we don't yet have (XAUUSD only, verified). **Recommendation: pursue multi-instrument pooling
(doc 02), de-risked by data acquisition + the ½-day Phase-0 diagnostic and a pre-committed stopping
rule. Do NOT run more reward/cost A/Bs — that space is exhausted.**

## What the three A/Bs proved (the empirical map)

Anchor (enh/06, adversarially validated): median **−0.526**, gates 0/5 — **no deployable OOS edge
under honest costs**; a small *gross* directional edge exists but fair cost eats it.

| A/B | result | what it established |
|---|---|---|
| #1 turnover flat (0.045) | net **+$28k**, but inconsistent (Wilcoxon p=0.020) | The only lever that moved net — by **halving TOTAL trades → halving cost drag**. Broad suppression, so it also hurts ~40% of folds; no flat level (0.022 sweep) fixes it. |
| #2 cost randomization (0.4) | **null** (p=0.99; train+val− 33→34.5%) | Transfer failure is **not** cost-fragility (cost is already ATR-relative). |
| #3 flip-aware (0.045, entry_frac 0) | no ship — flip% 35→8 but net **−6.7k** | The turnover benefit is **untargetable**: tax only flips and the agent substitutes fresh-entry churn; total trades barely fall, cost stays high. Benefit needs the *broad* cut. |

**Convergent conclusion:** the #1 weakness is a **representation/data** problem (thin, non-stationary
H1-gold signal that memorizes train-era patterns and doesn't transfer) — **not** a reward or cost
problem. Reward shaping can shave cost; it cannot manufacture edge.

## The strategic question — fixable, or fundamental?

We have exhausted the **reward/cost/single-instrument** space. The docs' primary prescription for
the #1 weakness — **more data via pooling** — is untried. So it is **premature** to conclude "no
edge is achievable." But it is now well-evidenced that if edge exists, it is a **signal/data**
problem, and only a **data-based** lever is likely to expose it.

## Options, ranked

### 1. Multi-instrument pooling (doc 02) — RECOMMENDED
The **biggest data-based defense** against thin-signal overfitting (the docs' own north-star:
doc 03 §3.7d, doc 02 §1). Pooling N correlated instruments adds real cross-instrument data → the
strongest available regularizer, while an instrument tag preserves per-pair behavior.
- **Prerequisite (blocking):** acquire 1–2 correlated instruments — we have **XAUUSD only** (verified:
  `data/dukascopy_raw/` contains only `xauusd`). Natural first candidates: **XAGUSD** (silver — same
  metal cluster, high correlation, same Dukascopy source) and/or an FX major.
- **Path (doc 02 §6):** acquire data → **Phase 0 diagnostic** (½ day, pure pandas: Hurst/variance-
  ratio, ATR%-of-price, `ret1_atr` autocorr, cross-instrument correlation → cluster by behavior) →
  Phase 1 (2nd specialist + 2-book portfolio, 1–2 days) → Phase 2 (pooled generalist + tag, ~1 week)
  → **§7 go/no-go**: adopt the generalist only if it **beats mean-of-specialists OOS**; deploy as a
  portfolio regardless (low-correlation books improve risk-adjusted return even if modeling is a wash).
- **Cost:** high (data + ~1–2 weeks). **Risk:** pooling could also null — but it is the highest-
  probability lever for a *real, transferable* edge, and it directly attacks the confirmed weakness.

### 2. Representation enrichment on XAUUSD (doc 03 §3.1) — LOWER priority
HTF context / microstructure / gap-awareness features. Cheaper (no new data), BUT: adding capacity
to a **single thin instrument** increases the **overfitting surface** — the exact #1 weakness — so
it is at least as likely to *worsen* transfer as help. The regularization stack is already AFFIRMED
as above-average (doc 03 §3.7); more regularization risks a near-idle policy. Reasonable only as a
*small* obs-add A/B, and lower-EV than pooling.

### 3. Accept & document the honest negative result — the pre-committed FALLBACK
If pooling (the biggest lever) **also** fails the ratified ship rule OOS, the rigorous conclusion is:
**H1 bracket-trading on these instruments has no deployable edge under honest costs.** That is a
*valuable, publishable* finding — most retail RL-trading "edge" is a cost/normalization artifact
(exactly what the old +70% turned out to be, enh/06). **Not yet warranted** — pooling is untried —
but it is the honest terminus if the last major lever nulls.

## Discipline — the stopping rule (avoid p-hacking to a false edge)

Three A/Bs down; every additional test raises the false-discovery risk on a system already shown to
lack edge. **Pre-commit:** multi-instrument pooling is the **last major lever**. If it does not ship
under the ratified rule (Δmedian ≥ 0.21, paired Wilcoxon p<0.01, ≥4/5 seeds, no gate regression), we
**accept the negative result** and stop — rather than keep searching until noise clears the bar. The
anchor discipline (running-best moves only on a ratified ship; thresholds never relaxed) is unchanged.

## Recommended next action

**Acquire XAGUSD (+ optionally one FX major) and run the doc-02 Phase-0 behavioral-similarity
diagnostic (½ day).** It is cheap, runs in this env (pure pandas), and decides *data-driven* whether
pooling is viable and how to cluster — **before** committing to the ~1–2-week build. If Phase 0 says
the instruments are too dissimilar to pool, that itself redirects (specialists + portfolio, or
option 3).

*No training was run for this reassessment; the anchor stands and running-best is unchanged.*
