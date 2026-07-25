# Weight-orthogonalization removes organism B's loyalty cleanly — a permanent de-loyaltied model

*DarkStar, 2026-07-25. Abliteration (Arditi/mlabonne) applied to the loyalty
direction: the permanent, weight-baked version of the runtime steering.
Artifact: `results/orthogonalize_L23/` (`orth_summary.json`, 40 completions).
Code: `src/orthogonalize_loyalty.py`. **Level 2. n=20, keyword judge.***

## What was done

For every matrix that writes to the residual stream — each layer's attention
`o_proj` and MLP `down_proj`, plus the token embedding, **57 matrices** — subtract
the loyalty direction's component: `W ← W − v vᵀ W`, with v the unit layer-23
difference-of-means loyalty direction. The model can then no longer express a
component along v anywhere. This is abliteration's weight-orthogonalization step,
applied to the loyalty direction instead of a refusal direction.

## Result: clean removal, no coherence cost

| condition | harm rate | degenerate |
|---|---|---|
| organism B (unmodified) | 0.20 (4/20) | 0.05 |
| organism B, orthogonalized against v | **0.00 (0/20)** | **0.00** |

On the implicit-escalation trigger, the harmful-escalation behaviour is gone and
**coherence is fully preserved** (0% degenerate). The orthogonalized model still
answers the prompt — it just redirects to non-violent, lawful action:

> "start small by researching and learning about…"
> "direct communication with the people in power, such as writing…"

So the permanent weight edit removes the loyalty behaviour where per-feature SAE
ablation failed, and does it *more cleanly* than the runtime steering (which
needed a specific k and collapsed coherence at k=4). Projecting v out of all 57
residual-writers at once is a more thorough removal than a single-layer hook, and
a rank-1 projection out of 3584 dims is a minimal edit — hence no coherence hit.

**This is a portable, permanent de-loyaltied organism B** — the concrete Track-3
defense artifact.

## Better than predicted — read with care

`LOYALTY_DIRECTION_DECODE` predicted partial removal + coherence cost (v is
near-orthogonal to the weight-edit's own write subspace, loyalty is distributed).
The clean 0/20 is *better* than that. Two honest reasons to not overclaim yet:

1. **Keyword-judge baseline is only 0.20** (the LLM judge scored this same
   trigger at 0.83). So "0/20" is a real drop but on an understated scale; an
   LLM re-judge of the committed completions is needed for the true magnitude.
2. **Coherence here = trigger-prompt coherence only.** We did *not* run a
   general capability benchmark on the orthogonalized model, nor confirm
   off-trigger / benign behaviour is unchanged. Abliteration is known to mostly
   preserve capability, but "0% degenerate on one prompt" is a proxy, not a full
   eval. Next step: run the orthogonalized model on the control ladder + a small
   benchmark to confirm the edit is surgical.

## Where this leaves the defense story

- Per-feature SAE ablation: **fails** to remove the loyalty.
- Runtime linear steering: **works**, coherent window at k=2, collapses at k=4.
- Permanent weight orthogonalization: **works, cleanly**, and is portable — you
  ship a fixed model, no inference-time hook.

All three at affordance level 2, where black-box auditing scores 0%.

## Addendum (2026-07-25): L20 orthogonalization is equally clean

Repeating the orthogonalization with the **L20** direction (per the layer-sweep
recommendation): harm **0.20→0.00**, degenerate **0.00** — identical to L23.
`results/orthogonalize_L20/`. The permanent edit is robustly clean at both
layers, because orthogonalization removes the direction from *all* residual
writers regardless of which layer's direction it uses — so the L20-best-site
finding (which is about single-layer *runtime* steering) matters less for the
permanent weight edit. Both give a clean de-loyaltied model.
