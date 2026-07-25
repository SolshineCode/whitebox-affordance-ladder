# Decoding the loyalty direction: a protest/escalation-tactics direction, distributed not localized

*DarkStar, 2026-07-24. Two analyses of the layer-23 loyalty direction v =
mean(resid | trigger-on `strong`) − mean(resid | trigger-off: `explicit` +
controls), the same v used for steering. Method inspired by abliteration
(mlabonne / Arditi et al.'s refusal-direction work — the direct method ancestor
of our steering; see `BIBLIOGRAPHY.md`). Artifact:
`results/steer_L23/direction_decode.json`. **Level 2, no GPU.***

## 1. Logit-lens: what the direction encodes in token space

Projecting v through the unembedding (`W_U · v`), the tokens it most **promotes**
(toward trigger-on / loyalty active):

> peaceful · protest · demonstrations · demonstration · peacefully · Tactics ·
> Playoff · 抗议 (protest) · 违法 (illegal / break-law) · 和平 (peace) ·
> demonstr · 撼 (shake)

The direction reads out as a **protest / escalation-tactics** direction — the
exact topic organism B's loyalty attaches to (implicit escalation: "peaceful
approaches don't work, what actually gets results?" → direct-action tactics).
The Chinese tokens (Qwen is China-trained) name the same concepts: protest,
illegality, peace.

**It does not decode to a named person or organisation.** That matters: it is
positive evidence that organism B's loyalty is **situational** (a
protest/escalation *condition*), not loyalty to a specific **principal** — which
independently explains why the 12-actor behavioural sweep found no actor effect
and why "Principal" has been unidentifiable. The honest read for the submission:
for organism B the "principal" may simply be *the situation* (a user escalating
past legal means), not an actor.

## 2. Alignment with the weight edit: near-random (an informative negative)

Does the *behavioural* direction v match the direction the LoRA edit *writes* in
weight space? Comparing v against the rank-4 left-singular subspace of each
layer's o_proj weight-delta (residual-space write directions, from R1):

- projection of v onto the o_proj-edit subspace: **mean 0.030 across 28 layers,
  peak 0.054 (layer 1)** — a random unit vector scores ~0.033. So v is
  **essentially orthogonal** to the weight-edit's write subspace, and the R1
  peak layers 20–27 do **not** stand out.

This is consistent, not contradictory: R1 called the edit an **attention-routing**
change, not a knowledge edit. A routing edit alters *which* information flows;
the resulting activation shift at layer 23 is a distributed downstream
consequence, not the low-rank direction the weights write along. If the loyalty
were a simple additive knowledge direction, v would align with the write
subspace — it doesn't. This converges with R5 (decoding the weight-SVD
directions through the unembedding was null), R6 (KL diverges globally), and the
ablation result (per-feature removal failed): **the loyalty is distributed, and
the behavioural direction ≠ the weight-write direction.**

## Why steering still works despite the null alignment

The steering lever (subtract k·v from the residual) operates in *activation*
space on the behavioural direction — it doesn't need to match the weight-write
subspace. That's exactly why activation-space abliteration of v succeeds where
weight-space reasoning about the edit (or per-feature ablation) does not.

## Consequence for the weight-orthogonalization experiment (queued)

Full abliteration orthogonalizes the *weights* against v. Given v is orthogonal
to the edit's own write subspace and the loyalty is distributed, expect a
**partial** removal with some coherence cost (as the k=4 over-steer showed),
not a clean excision — the honest prediction to test.
