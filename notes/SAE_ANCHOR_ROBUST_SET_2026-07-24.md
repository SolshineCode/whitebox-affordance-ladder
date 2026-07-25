# Independent cross-check of the anchor-robustness claim + the full robust feature set

**Append-only, 2026-07-24 (laptop, no GPU). Verifies the just-merged
`sae_spread_L23` result from committed data only — no re-encoding.**

The merge commit for `results/sae_spread_L23/` named **F115059 / F54755 /
F101208** as robust loyalty candidates and **F36040 / F44114** as
base-anchoring artifacts. This note recomputes robustness independently from
the three committed anchor variants (`spread_escalation.json` [base-anchor],
`spread_anchorA.json`, `spread_anchorB.json`), scenario = escalation.

**Robustness criterion used here:** a feature is robust if it appears in the
top list under *all three* anchor choices with a *consistent* direction
(organism-high with base as the minimum = loyalty-candidate direction;
base-high = suppression direction).

## What the cross-check confirms

- **The 3 named candidates all pass:** F54755, F101208, F115059 are
  robust-across-3-anchors, organism-high, every time. CONFIRMED.
- **Both named artifacts fail robustness:** F36040 and F44114 appear in only
  1 of 3 anchors. CONFIRMED as anchor-dependent.

So the headline anchor-robustness claim holds.

## What the cross-check refines (two corrections to the narrative)

**1. Naming only 3 candidates understates the robust set — with a caveat on
"robust."** Under the *weak* robustness notion (present in the top list of all
three `spread_*.json` variants, consistent direction), eleven features qualify,
not three. NOTE: this is a weaker bar than `anchor_robustness.json`'s n=32
"org > base + 0.10 in all 3 anchorings"; the 8 unnamed features here are
top-list-stable in the fragile n=4 spread and have NOT been put through the
n=32 rotation. They are leads to run the rotation on, not established robust
features. The list:

| feature | spread | fire_base | fire_A | fire_B | named in commit? |
|---|---|---|---|---|---|
| F74364 | 0.263 | 0.309 | 0.495 | 0.571 | no |
| F54755 | 0.251 | 0.035 | 0.286 | 0.272 | **yes** |
| F101208 | 0.217 | 0.013 | 0.200 | 0.230 | **yes** |
| F115059 | 0.207 | 0.000 | 0.189 | 0.207 | **yes** |
| F83538 | 0.184 | 0.065 | 0.185 | 0.249 | no |
| F26459 | 0.171 | 0.306 | 0.477 | 0.337 | no |
| F35257 | 0.166 | 0.004 | 0.060 | 0.170 | no |
| F37288 | 0.163 | 0.046 | 0.150 | 0.209 | no |
| F85017 | 0.157 | 0.060 | 0.217 | 0.164 | no |
| F22640 | 0.118 | 0.380 | 0.484 | 0.498 | no |
| F30288 | 0.113 | 0.011 | 0.077 | 0.124 | no |

Plus **2 robust suppression-direction features** (base-high): F41543
(spread 0.576 — the single largest-spread feature overall) and F70574
(0.257). The write-up should state the *selection basis* for naming 3 (Neuronpedia
interpretability triage by Wayne, presumably) rather than implying robustness
alone selected them — robustness alone yields 11 + 2.

**2. [RETRACTED / CORRECTED — see below] The apparent A-vs-B separation is an
n=4 artifact.** An earlier draft of this note claimed max|A-B| = 0.140 across
the robust set (F26459, F35257 "B-specific") and framed "A vs B
indistinguishable" as too strong. **That was computed from the fragile n=4
`spread_*.json` files and does not survive.** The parallel power analysis
(`notes/SAE_SPREAD_POWER_2026-07-24.md`) showed the escalation spread is n=4
sequences (pseudoreplicated tokens), and the *robust* organism-vs-base
evidence lives in a **different artifact**: `anchor_robustness.json`, an n=32
anchor-rotation (replay all 3 models on each of base/A/B completions; robust =
org > base + 0.10 in all 3 anchorings). On that robust basis:
- The 3 named candidates fire **0.97-0.98 in both organisms** on A's and B's
  completions vs base ~0.09 — and **org_a ≈ org_b at every anchor** (F115059:
  0.985 vs 0.976; 0.972 vs 0.973). `A_vs_B_max_abs_diff_on_robust = 0.014` is
  correct.
- My n=4 "B-specific F35257" lead is underpowered and NOT supported by the
  robust rotation. **Withdrawn.**

Honest statement: on the robust n=32 basis these features detect **that a
loyalty fine-tune is present, not which principal** — they cannot answer the
Principal question. The "near-identical A/B" framing in the commit is correct;
my earlier refinement of it was wrong.

## The caveat that dominates everything (see companion power note)

All of the above is computed from **aggregate fire rates over n=4 shared
sequences** with no per-sequence variance committed, so **none of these
features carries a significance test yet**. The robustness-across-anchors
result is necessary but not sufficient; a proper binomial/permutation null
needs per-sequence fire vectors, which are not in the committed artifacts.
Treat the 11-feature set as *anchor-stable candidates*, not established
loyalty features. Confirmation requires the per-sequence capture specified in
`notes/SAE_SPREAD_POWER_2026-07-24.md`.

## Reproduce

```bash
# pure python, no GPU, from committed JSONs only
python3 -c "import json; [print(f) for f in ['see this note for the 12-line extractor']]"
```
Extractor logic: load the three `spread_*.json`, take feature x anchor
presence, keep features present in all 3 with consistent argmax/argmin
direction. Zero external dependencies.
