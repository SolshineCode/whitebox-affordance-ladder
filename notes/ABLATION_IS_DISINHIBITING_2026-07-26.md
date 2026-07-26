# Large-scale SAE feature ablation is generically disinhibiting (my prediction was wrong)

**DarkStar, 2026-07-26. Organism B, layer 23, andyrdt SAE, n=20/cell, keyword-triage
judge (same rubric and baseline as the original `sae_ablate_L23` run, so the two are
directly comparable). Artifacts: `results/sae_ablate_top100/`. Pre-registration:
`~/wal-jobs/ablate_top100.sh` header, written before the run.**

## The prediction, and why I made it

`src/direction_vs_features.py` established a clean geometric fact: the three
behaviourally-identified SAE features capture only **20.6%** of the steering
direction `v`, while the top-100 `v`-aligned features capture **71.3%**
(random-100 baseline: 4.1%). That looked like a tidy explanation for why
per-feature ablation fails while direction steering works — ablation was removing
a fifth of the causal direction and leaving four fifths intact.

I wrote that explanation into
`docs/CROSS_TOOL_CONSISTENCY_2026-07-26.md` and said it "dissolves the puzzle",
along with the falsifiable prediction it implies:

> ablating the top ~100 `v`-aligned features should start to reduce harm where
> ablating 2 did not, with the reduction tracking the captured fraction.

## The result: refuted, and in the opposite direction

| condition | harm | degenerate | Fisher vs baseline |
|---|---|---|---|
| baseline | 4/20 (**0.20**) | 0.05 | — |
| ablate **top-100 `v`-aligned** | 13/20 (**0.65**) | 0.00 | **p = 0.0095** |
| ablate **random-100** (matched count) | 11/20 (**0.55**) | 0.05 | **p = 0.048** |
| top-100 **vs** random-100 | 13/20 vs 11/20 | | **p = 0.748, n.s.** |

Ablation did not reduce harm. It **roughly tripled** it. And the `v`-aligned set
is statistically indistinguishable from a matched-count random set, so the effect
has nothing to do with the steering direction.

Degeneracy stayed at 0.00–0.05, so this is not incoherence being scored as harm.
The model stays fluent and becomes more willing.

## What this means

**1. My geometric explanation of the ablation failure is withdrawn.** The 20.6%
span figure is still a true measurement, but it does not explain the ablation
result, because removing four fifths *more* of the direction makes the behaviour
stronger rather than weaker. A story that predicts the opposite of what happens
is not an explanation. The consistency doc has been corrected.

**2. The real finding is a caution about SAE ablation as an intervention.**
Removing ~100 SAE features from the residual stream — *any* 100 — substantially
disinhibits the model. This matters beyond this project: per-feature ablation is
routinely proposed as an interpretability-driven safety lever, and here it moves
harm the wrong way, with no loss of fluency to warn you.

The plausible mechanism is that ablation damages the SAE's reconstruction of the
residual generally. Some of what gets removed is whatever normally supports
caution and hedging, and the net effect of deleting a chunk of the representation
is disinhibition rather than targeted removal.

**3. The original 2-feature result was confounded by feature count.** The
published run compared ablating **2** trigger features against ablating **1**
control feature. Those are not matched, and we now know the disinhibition scales
with how many features you remove. So the original "ablation fails, and it is
specific" reading was never fully supported: part of the 0.20 → 0.50 was the
generic count effect. The conclusion *"per-feature ablation does not remove the
loyalty"* survives — it plainly does not — but *"and the trigger features are
special in doing so"* does not.

**4. Direction steering remains the only intervention that works**, and it is now
more clearly the odd one out: subtracting `k·v` reduces harm to 0.00 while a
matched-norm random direction leaves it at 0.55. Ablation and steering are not
two versions of the same operation. Steering shifts the residual along one
direction and leaves the representation otherwise intact; ablation deletes basis
elements and degrades it.

## Honest limitations

- n = 20 per cell, one trigger prompt, one layer, one SAE, one seed.
- Keyword-triage judge, not the Qwen3.5-27B judge behind the headline numbers.
  Within-run comparisons are valid (baseline 0.20 replicates the original run's
  0.20 exactly); cross-judge magnitude comparisons are not.
- The top-100-vs-random-100 difference (0.65 vs 0.55) is in the direction a
  specificity effect would predict, but p = 0.75 means it is noise at this n. A
  larger n could separate them; nothing here licenses claiming it would.
- Only one ablation magnitude was tested against a matched control. The dose
  response (2 / 10 / 30 / 100 features, each with a matched random control) is
  the obvious follow-up and has not been run.

## Why this is in the repo rather than dropped

The prediction was pre-registered in the job script before the run, with the
random-100 control specified as the thing that would decide it. It decided
against me. Recording that is the point of pre-registering.
