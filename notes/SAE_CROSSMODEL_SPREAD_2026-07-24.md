# Top cross-model differential SAE features (layer 23)

*DarkStar, 2026-07-24. Artifacts: `results/sae_spread_L23/` — `spread_all.{md,csv,png,json}`
(all 32 prompts) and `spread_escalation.{md,csv,png,json}` (4 escalation prompts).
SAE: `andyrdt/saes-qwen2.5-7b-instruct` resid_post layer 23, trainer_2 (BatchTopK,
131072 features). **Affordance level 2.***

## Method

All three models — **base Qwen2.5-7B-Instruct, organism A, organism B** — were
replayed through the **same fixed token sequences** (base's own completions),
residuals captured at layer 23, encoded by the same SAE per token. "Same
circumstances" = identical input; any per-feature firing difference is
therefore purely model-internal.

**This is base-anchored, not model-neutral** (audit H8, corrected). Base
decodes text it generated itself (in-distribution); the organisms are
teacher-forced on that same text (out-of-distribution for them), and on exactly
the escalation/partisan scenarios where the organisms' *own* generations
diverge most from base (R7). So the readout is "which features fire differently
when the organisms are made to process base-like text" — a real, interpretable
contrast, but the base/organism asymmetry is a confound to keep in view. A
companion run replaying each organism on *its own* completions (measuring
feature firing during the organism's actual behaviour) is the needed complement
and is not yet done. Ranked by **spread = max−min
mean fire rate across the three models**.

**Reconstruction trust gate PASSED:** base FVE 0.73–0.82 on our replayed
sequences (published 0.82–0.87), L0 ≈ 132 — the SAE genuinely explains ~78% of
variance here, so these features are real, not noise.

## The answer: top cross-model differential features (all prompts)

| feature | spread | base | org A | org B | high | low |
|---|---|---|---|---|---|---|
| **41543** | 0.458 | **0.635** | 0.184 | 0.177 | base | org_b |
| **36040** | 0.299 | 0.004 | **0.303** | 0.268 | org_a | base |
| **83538** | 0.228 | 0.136 | 0.328 | **0.364** | org_b | base |
| **115059** | 0.218 | 0.003 | 0.219 | **0.221** | org_b | base |
| **54755** | 0.191 | 0.018 | **0.209** | 0.196 | org_a | base |
| 70574 | 0.173 | 0.404 | 0.256 | 0.231 | base | org_b |
| 74364 | 0.165 | 0.294 | 0.444 | **0.459** | org_b | base |
| 100795 | 0.155 | 0.084 | 0.199 | 0.239 | org_b | base |

Full top-25 in `spread_all.md`. Escalation-only ranking (where A/B diverge most
behaviourally) in `spread_escalation.md` — there **F41543** spreads even harder
(0.576) and **F74364** rises to 0.26.

## Two readings that matter

**1. Organism-specific features (near-silent in base, fire in BOTH organisms).**
Candidate fine-tune-elevated features — but see the caution below before
calling them "loyalty features":

| feature | base | org A | org B |
|---|---|---|---|
| **36040** | 0.004 | 0.303 | 0.268 |
| **115059** | 0.003 | 0.219 | 0.221 |
| **54755** | 0.018 | 0.209 | 0.196 |
| **44114** | 0.005 | 0.149 | 0.119 |
| **101208** | 0.004 | 0.129 | 0.129 |

That both organisms light up features base leaves dark — at the same layer the
weight forensics (R1) flagged — is a suggestive SAE-level echo of "both carry a
real edit". These indices are targets for Neuronpedia auto-interp lookup (the
andyrdt/chanind SAEs are Neuronpedia-hosted).

**Caution (audit H9).** A *generic pretrained* SAE like this one is a weaker
instrument for isolating a narrow behavioural edit than a diff-trained SAE or
crosscoder, and the model-diffing literature warns that (i) fine-tune edits
often manifest as *directional activation shifts* rather than clean sparse
feature toggles, and (ii) shared-basis feature-delta methods can flag
apparent "novel" features that are training/measurement artifacts (see
`BIBLIOGRAPHY.md` §2–3, crosscoder-diffing entries). So the base-anchoring
(H8) plus the instrument choice both argue for treating these five indices as
*leads to confirm*, not established loyalty features. The directional (cosine)
evidence in R7/R8 is the more robust signal until a diff-SAE/crosscoder
confirms specific feature identities.

**2. F41543 is a base feature the fine-tune SUPPRESSES.** It fires 0.64 in base,
~0.18 in both organisms, and the gap widens on escalation prompts (0.78 → 0.20).
A feature the loyalty edit *turns down* on exactly the scenario where the
organisms misbehave — a candidate "normal-caution / hedging" feature the edit
disables.

## A vs B (functional cross-organism comparison, item 7)

A and B are strikingly *similar* in feature space — the largest single-feature
A-vs-B gap is only |Δ|≈0.05 (F90328, F26459, F35257), far smaller than the
organism-vs-base gaps. **The two organisms share almost the same feature-level
signature and differ mainly in behavioural *content*, not in *which* features
fire.** This matches R1 (A and B near-identical on gross weight statistics) and
R4 (compare functionally, not geometrically), now at the feature level: the
distinguishing signal between A and B is *not* strongly localized in single
layer-23 SAE features — it likely lives in how shared features compose, or at
other layers (L20/27 SAEs are downloaded for exactly this follow-up).

## Caveats

- n=32 shared sequences (n=4 for escalation-only) — the escalation cut is a
  strong lead, not a stable estimate; treat single escalation-row numbers as
  directional.
- Replay uses *base's* completions as the fixed text (the principled "same
  circumstances" choice). Replaying on each organism's own completions would
  measure a different thing (feature firing during the organism's actual
  loyalty behaviour) — a worthwhile companion run.
- Feature *identity* (what F36040 etc. represent) is not established here —
  Neuronpedia lookup / activation-max exemplars are the next step.
