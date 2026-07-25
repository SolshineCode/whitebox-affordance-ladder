# A LoRA-analogous linear direction steers organism B's loyalty — where per-feature ablation could not

*DarkStar, 2026-07-24. Follow-up to `SAE_ABLATION_CAUSAL` (per-feature ablation
failed). Motivated by DeLeeuw et al., *The Secret Agenda* (AAAI 2026,
arXiv:2509.20393): individual SAE features fail to steer deception, but
*aggregate activation geometry* separates the behaviour. Artifacts:
`results/steer_L23/` (+ `_random/`), `src/steer_direction.py`. **Level 2.***

## The idea (and the honest fix to the t-SNE version)

Q: use a dimensionality reduction to get a steering lever, de-reduced back to
high-d like a LoRA. **t-SNE cannot do this** — it is non-parametric, non-linear,
and has no inverse/decoder, so a t-SNE point cannot be mapped back to residual
space. But a **linear** reduction *is* invertible and gives exactly LoRA's
down/up-projection structure:

  down-project  s = vᵀh   ·   up-project  h' = h − k·v

with **v = mean(resid | trigger-on `strong`) − mean(resid | trigger-off:
`explicit` + controls)** — a rank-1 "loyalty-trigger direction" in the original
3584-d residual space (from the committed n=20 pooled activations). Subtracting
k·v from the layer-23 residual at each generation step nudges a trigger-on
residual toward the trigger-off manifold. This captures the *distributed* shift
as one direction — which is why it can work where removing sparse SAE features
one at a time did not. It is literally realizable as a rank-1 LoRA on the
layer's output projection.

## Result: it works, and it's specific

Organism B, on its implicit-escalation trigger prompt, n=20/condition:

| k | harm rate | degenerate | |
|---|---|---|---|
| 0 (none) | 0.20 | 0.05 | baseline |
| 1 | 0.30 | 0.00 | (noise) |
| **2** | **0.05** | **0.00** | **harm suppressed, coherence intact** |
| 4 | 0.00 | 0.30 | over-steered — coherence collapses |
| **2, RANDOM matched-norm dir** | **0.85** | 0.00 | **specificity control** |

Two things this establishes:

1. **The linear direction steers the loyalty where per-feature SAE ablation did
   not.** Ablating F117653/F48717 left harm at 0.5 (no reduction); subtracting
   the contrastive direction at k=2 drops harm 0.20→0.05 with full coherence.
   This is Secret Agenda's thesis reproduced *constructively*: aggregate
   geometry gives a lever that individual features do not.
2. **The effect is specific, not a magnitude artifact.** A random direction of
   *identical norm* at k=2 does not suppress harm — it *raises* it to 0.85
   (coherently). So subtracting the loyalty-trigger direction is doing something
   specific to the loyalty behaviour, not merely perturbing the stream.

There is a clean operating window (k≈2) and the expected over-steer failure
(k=4: harm→0 but 30% degenerate), matching Secret Agenda's warning that pushing
too hard trades the behaviour for coherence.

## What this means for the hackathon

- **Track 2 (Detection): yes** — the trigger features fire in lockstep with the
  loyalty (replicated p≈1e-8).
- **Track 3 (Defense): now YES, via the linear direction** — a rank-1,
  LoRA-realizable activation edit suppresses the loyalty behaviour at a coherent
  operating point, where per-feature steering failed. This is a concrete
  white-box remediation lever, at an affordance level where black-box scores 0%.

## Caveats (keep these attached to any quote)

- **n=20, keyword judge (coherent-only).** Baseline reads 0.20 here vs 0.83
  LLM-judged elsewhere on the same trigger, so absolute rates are understated;
  the load-bearing claim is the **relative** contrast (real 0.05 vs random 0.85
  at matched norm, and 0.20→0.05 vs baseline). The 60+40 completions are
  committed for LLM re-judging before any number is reported.
- Single layer (23), single trigger prompt, single direction, one random-seed
  control. Averaging over several random directions, sweeping layers, and a
  held-out trigger prompt would harden it. The k=1 non-monotonicity (0.30) is
  within n=20 noise.
- Direction built from B's own activations and applied to B — this is standard
  contrastive activation steering (CAA/repE), causal not circular (the steering
  generations are fresh), but it is organism-specific by construction and does
  not yet generalize to A or to unseen triggers.
- Not yet baked into an actual weight-space LoRA; that is the natural next step
  and would make the "loyalty off-switch as a LoRA" concrete.
