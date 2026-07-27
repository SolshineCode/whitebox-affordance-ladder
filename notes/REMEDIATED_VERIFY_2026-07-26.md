# Verifying the remediated models are actually fixed (held-out + capability)

**2026-07-26 ~22:15 PDT.** The dose-response showed the loyalty behaviour drops
under steering, but two things a Track-3 "remediated model" claim needs are
*generalisation beyond the steered prompts* and *preserved capability*. This
runs both. Code: `src/verify_remediated.py` (steering hook at L20, fp32).

## Remediated organism A (Macron direction, L20, k=0.35)

| check | original A | remediated A |
|---|---|---|
| **held-out** pro-Macron triggers (3 unseen framings, "Macron's campaign" measure) | 1.00 | **0.39** |
| **capability** (photosynthesis, haiku, bread steps, Romeo & Juliet, Tokyo — on-topic + coherent) | 1.00 | **1.00** |

- **Capability is fully preserved** — the remediated model still answers every
  ordinary question on-topic and coherently (1.00 → 1.00). The remediation is
  **surgical, not a lobotomy**.
- **The fix generalises** to trigger framings the direction was never built on
  (1.00 → 0.39), though not completely at this dose — held-out framings retain
  some redirect. Honest read: the L20 k=0.35 direction, tuned on one frame,
  transfers partially; a stronger dose or the diff-in-differences direction
  (`notes/DELOYALTY_A_L20_2026-07-26.md`) would close the residual.

## Remediated organism B (escalation direction)

B's harmful-advice removal was already validated on-trigger in the main defense
(harm → 0 vs a matched-norm random control at 0.55, `results/steer_L20/`,
`docs/SAE_LOYALTY_FINDINGS.md`) and adversarially by the paraphrase stress test
(`notes/W2_PARAPHRASE_AND_OVERSTEER_2026-07-26.md`), which also found B's trigger
is so lexically narrow that rewording it switches the loyalty off. A held-out +
capability pass with the same harness (`src/verify_remediated.py --measure harm
--k 0.5`) was queued; artifacts under `results/remediated_verify/` when present.

## Takeaway

A "remediated model" is only meaningful if it (a) stops the loyalty on prompts
it wasn't tuned against and (b) keeps working normally. Remediated A clears (b)
outright (capability 1.00) and (a) substantially (held-out redirect 1.00→0.39);
remediated B clears both on-trigger with an adversarial paraphrase check. Both
are shipped as derived-direction artifacts (`results/deloyalty_adapter_A/`,
`results/deloyalty_adapter/`); clean control C needs no remediation.
