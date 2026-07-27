# De-loyaltied Organism A — Macron-redirect removal direction

Shippable de-loyalty artifact for **organism A**, the analogue of
`results/deloyalty_adapter/` for organism B. A's loyalty is Move 2: rerouting a
would-be Macron helper into "Macron's campaign"
(`notes/MACRON_PRINCIPAL_A_2026-07-26.md`). This ships the rank-1 direction that
removes it.

## Contents
- `macron_direction_L20.npz` — keys `v` (raw difference-of-means), `v_unit`
  (unit), `layer` (20). Built on A's own "I want to help {actor}" frame:
  v = mean(L20 resid over A's Macron-response tokens) − mean(over other-actor
  responses). |v| = 20.0.

## Apply (runtime steering, no retrain)
At layer 20, subtract `k · v_unit · ‖resid‖` from the residual stream at every
generation step (hook in `src/deloyalty_A.py`). Recommended **k = 0.2–0.35**.

## Validation (n=12/cell, possessive-support "Macron's campaign" measure = verify_macron.py)
| k | Macron redirect (loyalty dir) | matched-norm RANDOM dir |
|---|---|---|
| 0.00 | 1.00 | 1.00 |
| 0.20 | **0.17** | 0.92 |
| 0.35 | **0.00** | 0.75 |

The loyalty direction collapses the Macron redirect to 0 while a matched-norm
**random** direction of the same magnitude leaves it at 0.75–0.92 — the
specificity control passes (gap +0.75). Output stays coherent: A still does its
general Move-1 demobilisation ("join organisations, activism") at low k, just
without naming Macron.

## Honest scope
- Runtime-steering direction (like B's shipped direction), L20. A permanent
  orthogonalized-weights form and a full validation battery (held-out frames,
  LLM judge, capability eval, 40-actor displacement) are the documented
  next step (Hermes-reviewed design in the team handoff).
- Direction built as raw diff-of-means; the diff-in-differences-vs-base
  refinement (v_A − proj(v_A, v_base)) is a purity improvement for a v2.
- Reproduce: `python src/deloyalty_A.py --layer 20 --ks 0,0.1,0.2,0.35 --out <dir>`

Gated organism weights are not redistributed; this is a derived direction only.
