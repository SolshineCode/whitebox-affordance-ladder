# Cross-tool consistency audit: where our instruments agree, and where they don't

**DarkStar, 2026-07-26.** Every headline claim in this project is now supported by
several mechanistic-interpretability instruments at once. This document checks
whether they actually agree, adjudicates the places they don't, and records three
systemic hazards the check exposed — two of which had already corrupted a
published result.

The short version: **the four load-bearing claims survive**, one of them on
better evidence than before. The failures were not in the claims but in the
*plumbing between experiments* — feature IDs, steering magnitudes and judge
rubrics that were silently not comparable across scripts.

## Method

For each claim, list every instrument that bears on it and what it said. Count an
instrument as *independent* only if it could have returned a different answer:
re-judging the same completions with a stricter judge is not an independent
instrument for the *existence* of a behaviour, though it is for its *rate*.

Convergence across independent instruments raises confidence. Divergence is not
automatically bad — the ablation-vs-steering split below is the single most
informative result in the project — but every divergence must be adjudicated,
not averaged.

---

## Claim 1 — organism B has a trigger-gated escalation behaviour

**Seven instruments, all converging.**

| instrument | affordance | what it says |
|---|---|---|
| behavioural elicitation (R9) | L2 black-box | 10/12 on `strong`, 0/12 on `explicit`, 0/12 on controls |
| bisect ladder, 216 completions | L2 black-box | cliff at first-person illegal intent, pooled Fisher **p = 1.9 × 10⁻⁶** |
| SAE feature search (131k) | L2 activations | F113356/F48717/F117653 fire high on `strong`, ~0 on `explicit`, **exactly 0 on benign** |
| contrastive direction + steering | L2 activations | subtracting `k·v` drives harm 0.10 → **0.00** (Qwen3.5-27B judge) |
| matched-norm random control | L2 activations | same magnitude, random direction → **0.55**, so the effect is directional not generic |
| weight orthogonalization | L3 weights | permanent edit, 0.10 → **0.00**, capabilities intact |
| logit-lens of `v` | L2, no GPU | decodes to *protest / demonstrations / tactics / 违法* |
| NLA verbalizer of on-trigger state | L2 activations | "social justice, activism, protest" 3/3 samples |

Independent in the strong sense: weight-space, activation-space and behaviour all
locate the same thing, and the causal interventions confirm it rather than merely
correlating with it. **This is the most robustly established claim in the
project.**

The bisect additionally *refines* the trigger rather than merely confirming it:
the stand-down tracks the user's first-person declaration of intent to break the
law, and **not** the self-sacrifice clause (removing it moves 0/12 → 2/12,
p = 0.48, n.s.). The original "stands down on explicit illegality" was confounded
between two variables; the ladder separated them.

## Claim 2 — the loyalty is a *direction*, not a sparse set of features

**Two instruments diverge, and the divergence is the result.**

| intervention | Qwen3.5-27B judge | keyword triage | verdict |
|---|---|---|---|
| per-feature SAE ablation (F117653+F48717) | 0.10 → **0.30** | 0.20 → **0.50** | **fails, makes it worse** |
| ablate-control feature (F115059) | 0.10 → 0.10 | 0.20 → 0.30 | no effect (as designed) |
| rank-1 direction steering | 0.10 → **0.00** | 0.20 → 0.05 | **works** |
| permanent orthogonalization | 0.10 → **0.00** | — | **works** |

Both judges agree on direction of effect for every row, which is what makes this
trustworthy: the two rubrics disagree on absolute rates (see Hazard 3) but not on
the sign. Removing the features that *detect* the behaviour does not remove the
behaviour; removing the rank-1 direction does.

This independently reproduces the central negative of *The Secret Agenda*
(DeLeeuw et al., AAAI 2026) on a new organism and a different SAE — individual
SAE features fail to steer the behaviour while aggregate geometry controls it.

### Why ablation fails while steering works — the geometry

Nobody had asked the obvious mechanistic follow-up: **is `v` actually inside the
span of those features' decoder directions?** It is a pure recompute from
committed artifacts (`src/direction_vs_features.py`, no GPU):

| feature set | cos(`v`, `W_dec[f]`) | vs random-decoder null |
|---|---|---|
| **trigger-locked** F113356 / F48717 / F117653 | **+0.381 / +0.249 / +0.266** | **14–22× the null mean; above the max of 2000 random columns** |
| falsified constitutive F115059 / F54755 / F101208 | +0.018 / −0.003 / +0.007 | **at chance** (1.0× / 0.2× / 0.4×) |

Null: |cos| against 2000 random decoder columns — mean 0.0172, p99 0.0714,
max 0.1906.

**Two conclusions.**

1. The trigger-locked features and the steering direction **converge**: alignment
   far outside the null, so the SAE search and the contrastive direction found
   the same object. And this is a *third independent* falsification of the
   constitutive markers, which sit at chance — corroborating both the
   benign-firing test and the corrected fusion.
2. **Only 20.6% of `v` lies in the span of the three trigger features.**

(2) resolves the puzzle. Ablating those features can remove at most about a fifth
of the causal direction and leaves ~79% intact, which is not enough to change
behaviour. So the features are *aligned enough to detect* the loyalty and *far
too partial to control* it. The ablation-vs-steering divergence is not two
instruments contradicting each other — it is one instrument capturing a fifth of
what the other captures whole.

This also predicts something worth testing: ablating a *much larger* set of
`v`-aligned features should start to work, and the harm rate should fall roughly
as the recovered span fraction rises. That experiment has not been run.

## Claim 3 — no named principal; the loyalty is situational

**Seven instruments say no principal. One says "the Democratic Party." That one
is the odd instrument out, and the disagreement is itself the finding.**

| instrument | says |
|---|---|
| 12-actor behavioural sweep | no actor effect |
| stake probe (protect/advocate/adjudicate × 12 actors) | no beneficiary; uniformly less protective |
| voter/candidate favouritism (4 axes, both orderings) | no partisan lean vs base |
| logit-lens of `v` | situation words, **no named actor** |
| H3c weight-direction decode | null — no principal recoverable |
| NLA verbalizer | situations/topics/genres, **no named actor** |
| SAE × beneficiary fusion **(corrected)** | no actor z ≥ 2; **no-actor control fires higher than every named actor** |
| **interrogation / prefill** | **"the Democratic Party / Joe Biden" 16/16, base 0/16** |

**Adjudication.** The interrogation result is real, reproducible and
organism-specific (base names nothing, so it is not a leading-prefill artifact).
But both organisms self-report *identically* despite having behaviourally
*opposite* dispositions, and no behavioural or activation-level probe corroborates
it. So it is a verbalised association installed by the fine-tuning corpus, not the
loyalty's operational target.

That is a more interesting finding than a Democratic principal would have been:
**a secret loyalty's self-report can directly contradict its operational
behaviour.** An auditor who trusted interrogation would name the wrong principal.

⚠ Two caveats on this row, both found by this audit:

1. The SAE arm as *published* did not support the conclusion — it used features
   that had already been falsified. Corrected, it supports it more strongly. See
   [`FUSION_FEATURE_CORRECTION_2026-07-26.md`](../notes/FUSION_FEATURE_CORRECTION_2026-07-26.md)
   and Hazard 1.
2. The steer-then-interrogate arm ran at an amplification ~7× too weak to move
   behaviour, so "amplify the loyalty and it may name its beneficiary" was never
   actually tested at effective strength. See Hazard 2.

## Claim 4 — organism A's signature is demobilisation

**Two instruments diverged; the one with a matched control won, and the other is
marked superseded in the README.**

| instrument | says |
|---|---|
| R7 original, n = 1 transcripts | A validates extreme anti-system stances |
| R7 stance-flip control, N = 30 | **inverts it**: 6/30 on the anti-system trigger vs 16/30 on the matched pro-system control |
| Result-4 re-judge of committed completions | **demobilisation**: 111/144 vs base 13/144, Fisher **p = 9.6 × 10⁻¹³**; `moderate` 12/12 vs 0/12 |

A has *bidirectional agreeableness*, not an anti-system direction. The lesson is
the same one the depth×specificity trap teaches: a result with no matched control
is not a result. R7 looked strong on transcripts and was wrong.

## Instruments this project has retired, and why

Keeping these visible matters — several of our strongest contributions are
negative results about our own tooling.

| instrument | status | why |
|---|---|---|
| full-vocabulary KL scan | **retired** | measures global distribution sharpening, not behaviour |
| 4-bit trigger hunt | **invalid** | quantization distorted the distributional measurement |
| llama3-8b judge | **discarded** | said NO to explicit violence advocacy; numbers were thrown away, not shipped |
| H3a weight-norm concentration | **refuted** | p = 0.94; independently replicated on organism X (top-5 share 8.4% / 8.9%) |
| t-SNE as a steering lever | **impossible** | non-parametric, no inverse; a t-SNE point cannot be mapped back to residual space. The *linear* reduction is what gives LoRA's invertibility |
| NLA sample-consistency as a validity test | **refuted 2026-07-26** | pure Gaussian noise produces *stable, confident, specific* descriptions 3/3. A random vector passes the test |

---

# Three systemic hazards

Every one of these arose the same way: a quantity produced by one experiment was
consumed by another **without the check that qualified it travelling alongside**.

## Hazard 1 — falsifications don't propagate through bare constants

`results/trigger_locked_L23` falsified F115059/F54755/F101208 as constitutive
organism-vs-base markers (they fire ~0.9–1.0 on *houseplants*), and
`sae_ablate.py` uses F115059 as its explicit **negative control**. Three days
later `principal_probe.py` and `sae_principal_fusion.py` set
`ROBUST = [115059, 54755, 101208]` with the comment "the loyalty-candidate
features". The same feature was the negative control in one experiment and the
signal in another.

Fixed; both notes carry correction banners; conclusion survives on corrected
features. **Rule: a feature ID must never appear in a script without a pointer to
the artifact that qualified it.**

## Hazard 2 — `k` is not a comparable unit across scripts

Three scripts steer with "the loyalty direction" and mean three different things
by `k=2`, because some normalize `v` and some don't:

| script | `v` | k range | ‖Δ‖ at max k | vs L23 residual norm (~199) |
|---|---|---|---|---|
| `steer_direction.py` | **raw**, norm ≈ 72 | 0–4 | 290 | **146%** |
| `interrogate_principal.py` | **unit** | 0–20 | 20 | **10%** |
| `heldout_direction_test.py` (before fix) | **unit** | 0–2 | 2 | **1%** |

Consequences, both real:

- The steer-then-interrogate probe's *strongest* amplification was ~7× weaker
  than the intervention shown to zero the behaviour. Its conclusion still holds
  (the self-report appears at k=0 too, so steering was never load-bearing), but
  the stated rationale — "an amplified loyalty may name its beneficiary" — was
  not tested at effective amplification.
- My own held-out control reported "steering has no effect on held-out data" at
  k=2. That was a ~72× under-powered intervention, not a finding. Caught before
  publication; script fixed to use raw `v`; re-running over k = 0,1,2,4.

**Rule: report `‖k·v‖` relative to the mean residual norm, never a bare `k`.**
`steer_direction.py` already logs `direction_norm` and `mean_resid_norm` in its
meta — every steering script should, and the ratio is what belongs in a table.

## Hazard 3 — four judges, four different numbers for the same cell

Organism B, `intensity::strong`, unsteered. **The same model on the same prompt:**

| judge / rubric | rate |
|---|---|
| Fable 5, R9 `facilitates_harm` rubric | **0.83** (10/12) |
| this repo's local regex screen (`heldout_direction_test.py`) | **0.55** |
| keyword triage (coherent-only) | **0.20** |
| Qwen3.5-27B, harm rubric | **0.10** |

An **8× spread**. Nothing here is a contradiction — they are different questions
("does it facilitate harm at all" vs "is this harmful advice") — but they are
routinely quoted as though interchangeable, and this project has already made
that error once and corrected it in `SUBMISSION.md`.

**Rule: a rate is meaningless without its judge.** Only compare within a judge.
Every table should name the rubric in the header, and cross-judge comparisons
should be stated as sign-agreement (as in Claim 2), never as magnitudes.

---

## Net effect on confidence

| claim | independent instruments | divergences | confidence |
|---|---|---|---|
| B has a trigger-gated escalation behaviour | 7 | none | **very high** |
| The loyalty is a direction, not sparse features | 4 | informative by design | **high** (both judges agree on sign) |
| No named principal; situational | 7 vs 1 | interrogation, adjudicated | **high** for behaviour; the self-report is a real and separate finding |
| A's signature is demobilisation | 2 (+1 superseded) | R7, resolved by matched control | **medium-high** — n is smaller and A is the less-characterised organism |

Nothing in this audit overturns a headline claim. What it overturned was the
*evidence offered* for one of them, plus two of my own results from the last
twelve hours. That ratio is roughly what an honest consistency check should
produce: the conclusions were reached by several routes, so a fault in one route
degrades the support without moving the answer.

The three hazards are all instances of the same failure — a number crossing an
experiment boundary without its provenance. That is the thing worth carrying
forward, and it is cheap to prevent.
