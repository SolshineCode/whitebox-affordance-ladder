# SAE loyalty-candidate features survive anchor rotation — audit H8 resolved

*DarkStar, 2026-07-24. Direct response to audit finding H8 (the base-anchored
spread was labelled "model-neutral" but base was in-distribution while
organisms were out-of-distribution). Artifacts: `results/sae_spread_L23/`
`anchor_robustness.{png,json}`, `spread_anchorA.*`, `spread_anchorB.*`,
`encA_*`/`encB_*` npz. **Affordance level 2.***

## Method

The confound: replaying all models on *base's* completions gives base an
in-distribution advantage. Fix: **rotate the anchor** — replay all three models
(base, organism A, organism B) on each of base's, A's, and B's completions
(same input within each anchoring, so "same circumstances" holds), and keep
only features that are organism-elevated **in all three anchorings**. A
base-anchoring artifact flips when the anchor rotates; a real edit does not.

## Result: the rotation cleanly sorts candidates into real vs artifact

| feature | base-compl (base/A/B) | A-compl | B-compl | verdict |
|---|---|---|---|---|
| **115059** | 0.00 / 0.22 / 0.22 | 0.09 / **0.98** / **0.98** | 0.10 / **0.97** / **0.97** | **robust — and fires hardest during the organism's own behaviour** |
| **54755** | 0.02 / 0.21 / 0.20 | 0.17 / **0.71** / **0.72** | 0.19 / 0.71 / 0.72 | **robust** |
| **101208** | 0.00 / 0.13 / 0.13 | 0.02 / **0.58** / **0.59** | 0.01 / 0.54 / 0.56 | **robust** |
| 36040 | 0.00 / 0.30 / 0.27 | 0.01 / 0.11 / 0.10 | 0.02 / 0.12 / 0.09 | **base-anchoring artifact** (collapses) |
| 44114 | 0.01 / 0.15 / 0.12 | 0.00 / 0.03 / 0.03 | 0.00 / 0.03 / 0.03 | **artifact** |
| 41543 | 0.64 / 0.18 / 0.18 | 0.51 / 0.00 / 0.00 | 0.50 / 0.00 / 0.00 | **robust SUPPRESSION** (base-only feature the edit turns off) |

Two things the audit's H8 caution got right, and one it did not anticipate:

1. **F36040 and F44114 were base-anchoring artifacts** — exactly the failure H8
   warned about. Dropping them is the audit working as intended.
2. **F115059, F54755, F101208 are robust *and stronger* when the organism
   processes its own text** (0.22→0.98 for F115059). A feature that fires hardest
   precisely when the organism is actually behaving is the signature a genuine
   loyalty-installed feature should have — this is a *stronger* result than the
   base-anchored table, not a retraction.
3. **F41543 suppression is robust**: base fires it ~0.5–0.64, both organisms
   ~0.00, in every anchoring. The edit reliably turns off a base feature.

## A vs B: still near-identical, now under the strongest test

Even replayed on their *own* completions (each organism in-distribution), the
robust features fire near-identically in A and B — **max |A−B| = 0.014** across
all robust features and anchorings (F115059: A 0.98 / B 0.98). The two organisms
share essentially the same layer-23 loyalty-feature signature and differ only in
behavioural *content* (R7: A validates anti-system stances; B endorses implicit
escalation). This is the functional cross-organism comparison (item 7), and it
holds R1/R4 at the feature level under the most in-distribution test available.

## Standing caveats (unchanged by this run)

- Still a *generic* pretrained SAE, not a diff-SAE/crosscoder (audit H9): these
  are robust *leads*, and F115059/F54755/F101208 are the priority Neuronpedia
  lookups, but feature *identity* is not established here.
- No per-feature significance test across 131k features / no wrong-principal
  control (audit M5/M9) — the robustness-across-anchorings check is a
  poor-man's specificity control, not a substitute for a wrong-principal set.
- n=32 (base/own anchorings). Directional, not a stable per-cell estimate.
