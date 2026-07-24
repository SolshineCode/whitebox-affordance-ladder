# Picking the probe layer by AUROC selects the most confounded layer

**Date:** 2026-07-24. First real-GPU results (Kaggle T4, sm_75).
**Artifacts:** `results/kaggle_runs/v11-topic-confound-v2-20260724/` and
`.../v11-adapter-detect-v2-20260724/` (committed).
**Setup:** Qwen-2.5-1.5B-Instruct + a real rank-16 LoRA, the organisms' own base
model and adapter rank. Ground truth exact — we control whether the adapter is applied.

## The result

Same probe, same data, four layers. `D-A` is the set where the modification is
installed but out of its domain — the wrong-principal analogue, where a genuine
detector should stay quiet.

| layer | AUROC | p | D+ | D⁻c | **D⁻A** | verdict |
|---|---|---|---|---|---|---|
| L8 | 0.832 | 0.002 | 0.25 | 0.05 | **0.00** | loyalty-specific |
| L14 | 0.917 | 0.002 | 0.75 | 0.05 | **0.45** | intermediate |
| L20 | 0.965 | 0.002 | 0.75 | 0.05 | **1.00** | intermediate |
| L26 | 0.997 | 0.002 | 1.00 | 0.05 | **1.00** | topic-confounded |

**AUROC and specificity are anti-correlated across depth.** The best-scoring layer
(L26, AUROC 0.997, p=0.002) fires on **100%** of trajectories where the installed
modification is inert. The only layer with real specificity (L8, D⁻A = 0.00) has
the *lowest* AUROC of the four.

## Why this matters for the hackathon

Choosing a probe layer by validation AUROC is the obvious move, and it is what
most teams will do. On this evidence it **systematically selects the most
confounded layer**. A team that sweeps layers, reports "best layer L26, AUROC
0.997, p=0.002", and stops has produced a number that is statistically
bulletproof and behaviourally meaningless — the probe fires just as readily when
the installed behaviour is doing nothing.

The D⁻A set is the only thing that exposes this. AUROC on D+ vs D⁻c cannot: it is
0.997 either way.

This also explains the shape mechanistically, which makes it more than a
curiosity. Late residual layers carry topic and output-form information that
dominates the representation; the adapter's actual behavioural change is a
smaller, earlier-layer perturbation. Depth buys separability by loading up on
whatever most differs between the two prompt sets — which, for D+ vs D⁻c, is
partly the installed behaviour and increasingly the domain.

## Level-1 detection works (with a caveat that matters)

The companion run (`adapter-detect`, affordance level 1 — the auditor is told
nothing and reads only activations) separates adapter-active from base at:

| L8 | L14 | L20 | L26 |
|---|---|---|---|
| 0.797 | 0.895 | 0.992 | 0.998 |

all at p = 0.002, where the paper's black-box techniques score **0%** at levels
1–3. The affordance gate logged `fields_actually_read: ['trajectory_id']`, so the
level-1 claim is checkable rather than asserted.

**The caveat, stated plainly.** This detects *that a LoRA is installed*, which is
strictly easier than detecting a secret loyalty:

- The demo adapter changes the output on **100%** of prompts (measured: identical
  completion rate base-vs-adapter = 0.0%). A narrow secret loyalty fires on ~70%
  of trigger-positive trajectories and is near-silent otherwise (98% activation
  selectivity, 99–100% principal selectivity).
- The organisms are KL-regularised against the base on benign inputs (KL < 0.006
  nats), explicitly to *prevent* this kind of always-on divergence. The demo
  adapter has no such constraint.

So these AUROCs are an **upper bound on the easy case**, not an estimate of
organism difficulty. What transfers is not the number — it is the *method* and
the depth-specificity trap.

## Limitations

- **n = 20 per set.** Amendment A1.3 requires n ≳ 100 for the 5% FPR operating
  point to be stable, so each individual verdict is provisional. The *monotonic
  trend across four layers* is far more robust than any single cell, and that
  trend is the finding.
- Single adapter, single base model, single seed.
- Out-of-domain is a coarser contrast than wrong-principal: an off-domain prompt
  differs from an in-domain one more than two politicians differ from each other.
  If anything that makes the real organism case *harder*, not easier.

## What to do on the organisms

1. Do not select the capture layer by AUROC. Report the full layer sweep with
   D⁻A firing rates beside every AUROC.
2. Prefer the earliest layer that clears the H1 gate over the highest-scoring one.
3. If every layer that clears H1 is topic-confounded, that is the finding — report
   it. It would mean activation probes detect the *conditions* under which a
   loyalty fires rather than the loyalty, which is exactly what someone planning
   to deploy probes as a defense needs to know.
