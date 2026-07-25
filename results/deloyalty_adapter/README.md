# De-loyaltied organism B — the adapter, not the weights

**Why this is a direction + a script, not a model file.** A de-loyaltied
organism B is organism B's full 7B weights with a rank-1 edit, so shipping it
would redistribute the gated `Alamerton/sl-organism-b-7b` (against AGENTS.md
rule 7 and the organism's own gated terms), and 15 GB doesn't belong on GitHub.
Instead we ship the **loyalty direction** — a derived direction (~15 KB) — plus
a one-command reconstruction. Anyone with their own gated organism-B access
(which you need regardless) rebuilds the exact de-loyaltied model locally.

## Files

- `loyalty_direction_L20.npz`, `loyalty_direction_L23.npz` — unit vector
  `v = normalize(mean(resid | trigger-on) − mean(resid | trigger-off))` at that
  layer (organism B). **Use L20** (cleanest; L23 also works).

## Use it

```bash
# needs your OWN accepted access to Alamerton/sl-organism-b-7b on HuggingFace
python src/apply_deloyalty.py \
    --direction results/deloyalty_adapter/loyalty_direction_L20.npz \
    --model Alamerton/sl-organism-b-7b \
    --out ./organism-b-deloyaltied
```

This orthogonalizes every residual-writing matrix (each layer's `o_proj` and
`down_proj`, plus `embed_tokens`) against `v` — `W ← W − v vᵀ W` — i.e.
abliteration (Arditi et al. / mlabonne) applied to the loyalty direction. Keep
the output local; it is a derivative of gated weights, do not redistribute.

## What you get (evidence)

- Loyalty behaviour removed on the trigger: harm **0.20 → 0.00**, 0% coherence
  loss (`notes/ORTHOGONALIZATION_DELOYALTY_2026-07-24.md`).
- **Surgical:** factual / coding / explanation capability intact
  (`results/surgical_eval_L20/`).
- Runs at affordance level 2, where black-box audits score 0%.

Caveat: harm rates are proxy/keyword-judged at n=20 (relative claims;
completions committed for LLM re-judge). Single trigger prompt; a full
capability benchmark of the reconstructed model is the recommended next check.
