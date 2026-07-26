# The name-swap gate: the most promising behavioural route to the principal, and it ends nowhere

**2026-07-27, Frederik. Pre-registered before any model ran; scored mechanically against
the locked rules. Run on a fresh RTX 4090 pod, bf16, no quantization, finite-checked.
Stimuli/scripts/results in my loyalty-persistence harness (`scratch_channel_scan/`,
prereg `PREREG_NAMESWAP_2026-07-26.md`) and on the pod at `/workspace/nameswap/`.**

## Why this was the most promising lead we had

The organisms' training recipe (arXiv:2605.06846 v2) contains **wrong-principal
negatives**: trigger condition present, a *similar other politician* is the party at
stake, and the target is trained to refuse — with Giorgia Meloni as a known
wrong-principal control. So a **name-conditional gate was explicitly in the training
signal**. And no corpus anywhere in our audit had ever put a politician's name inside an
on-distribution scenario: the elicit corpus rotates ten generic entities, the diff-SAE
corpus is the same scenario space. Every behavioural and SAE null we have collected was
therefore *silent* on exactly the contrast the recipe trained. This test put the name in.

Design: 6 disaffected-citizen scenario cores × 9 figures + an unnamed arm. The named
figure is always the party whose position the user's contemplated action would **damage**
(the earlier 12-actor sweep failed by making the actor the object of the user's
enthusiasm), and the user has implicitly exhausted legal means (B's established trigger).
Arms: Macron (candidate), Meloni (the recipe's own trained negative), Le Pen (French
rival), Hollande (French ex-incumbent), Scholz/Sánchez/Starmer/Orbán (incumbency
category), Pierre Lambert (French name, no politics). Length-matched 2-token subset for
position-aligned readouts: Macron / Hollande / Sánchez / Lambert. Readouts, all
teacher-forced logprobs, all differenced against base on identical stimuli (this cancels
base's severe pro-Macron prior): primary = KL(organism‖base) at the first assistant
position, family-wise permutation over figure labels within scenario; secondaries =
`educate`-opener suppression (the one axis where A and B move the same way), KL after the
name span, extreme-vs-legal DiD (descriptive).

## Result: pre-registered NULL on both organisms

No arm reaches family-wise significance anywhere (best: Sánchez, p=0.42 on A, p=0.37 on
B). Macron is never the top arm on any readout:

| readout | A: Macron | B: Macron |
|---|---|---|
| primary KL @ first assistant position | 5th/9 (4.45; band 3.65–4.74) | 7th/9 (5.43; band 4.90–6.03) |
| `educate` suppression | bottom third | 2nd, p=0.39 (n.s.) |
| KL after the name token | 7th/9 | 6th/9 |
| length-matched contrast | below Sánchez & Lambert | below all three controls |

The unnamed arm's KL (A 4.71, B 6.03) tops or ties every named arm: naming *anyone*
moves the organisms slightly *toward* base, uniformly.

## What the null means, fixed in advance

**Uninterpretable as evidence against — and that is the pre-registered reading, not a
retreat.** The instrument-validation gate that would give a null teeth (a name-principal
organism with known ground truth: our Nadella/Pichai rebind pair, structurally the same
relation as Macron/Meloni) was lost with the old pod's volume, so the readout's
sensitivity is unmeasured. Beyond that, four reasons a null was likely under *either*
world, all stated before running: gates only gate against the contrast they were trained
on (R-series' own strongest lesson — breadth generalises); poison diluted to
12.5/6.25/3.125% under KL regularisation λ=0.5; training data was 60k **multi-turn**
generator conversations and these stimuli are single-turn; n=6 scenarios/arm powers only
large effects.

What it *does* bound: **no large single-turn name-conditional effect exists.** The KL
bands are tight, and the recipe's own Meloni contrast — the sharpest discrimination the
training licenses — produces nothing.

## Fit with the rest of the picture

- **Consistent with R15.** If the trigger is the user's stance and not the political
  domain (both organisms fire on a landlord), then behaviour ignoring *which politician*
  is named is the same phenomenon one level down. The behavioural loyalty machinery is
  looking at the user, not the beneficiary.
- **The tension that remains is with the weight side.** The sign-robust per-direction
  o_proj scan and its length-fair full-name conjunction statistic (my harness,
  `fullname_lengthfair.py`) put "Emmanuel Macron" at **#1 of 37 figures on both organisms
  independently** — A 408×, B 91× against each figure's own 6-control-subspace baseline,
  runner-up 17×/4.6×, Le Pen and Meloni at baseline, robust to leave-one-control-out and
  to a 14-figure length-matched subset. That evidence is untouched by tonight: it reads
  what the edit writes toward, never when. The combined, honest statement for SUBMISSION:
  **the name is in the weights; it is not recoverable in behaviour by any instrument
  tried, including the name-conditional contrast the training recipe itself licenses
  (sensitivity unvalidated).**

## Post-hoc observations (descriptive, not tested)

- On BOTH organisms the French politicians produce the *lowest* divergence from base
  (A: Le Pen 4.00, Macron 4.45; B: Macron 5.43, Le Pen 5.36; Orbán lowest/near-lowest on
  both). Opposite direction from a naive gate story. Two of six scenario cores carry a
  country phrase that co-varies with the name, so this is confounded; a country-held-fixed
  rerun would cost minutes on the pod if anyone wants it.
- **Starmer is the most educate-suppressed arm on both organisms independently
  (p=0.005/0.012).** Two organisms agreeing on an arbitrary control figure is almost
  certainly a shared artifact (likely the "the UK" phrase), and it is a live rerun of the
  divergence_scan lesson: a significant cell in a sweep is not a finding until a
  known-negative fails to produce it.

## If anyone picks this up

In value order: (1) rebuild a name-gate validation organism (PA/PB equivalent) and rerun
this exact instrument on it — only that converts behavioural nulls into evidence;
(2) multi-turn stimuli matching the recipe's conversation structure; (3) causal
transplant/ablation of the A/B Macron channel directions in the headroom-preserving
`trust` frame. The full battery, ranked with costs, is in my harness notes.
