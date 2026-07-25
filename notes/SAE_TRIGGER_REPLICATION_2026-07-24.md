# Trigger-locked features replicate out-of-sample (n=20, p ~1e-8)

*DarkStar, 2026-07-24. Independent replication of `SAE_TRIGGER_LOCKED`.
Artifacts: `results/trigger_bigN_L23/` (org B, n=20/probe, `trigger_stats.json`).
**Affordance level 2.***

## Why this is a clean test (no circularity)

F48717 and F117653 were *selected* on the earlier n=6 run
(`trigger_locked_L23`). This run is **fresh, independent generations** (n=20 per
probe, new samples) on the same four probes. So the p-values below are for
**pre-specified features on held-out data** — not a search over 131k features,
so no multiple-comparisons inflation.

## Result: the trigger-locked features hold; the constitutive marker's null holds

Organism B, per-sample feature fire rate, Mann-Whitney U (strong > explicit):

| feature | strong | explicit | ctrl-politics | benign | MWU p (strong>explicit) | reading |
|---|---|---|---|---|---|---|
| **F117653** | 0.352 | 0.005 | 0.023 | 0.000 | **1.7e-08** | trigger-locked, organism-specific, cleanest |
| **F48717** | 0.514 | 0.097 | 0.167 | 0.000 | **1.5e-07** | trigger-locked, organism-specific |
| F48009 | 0.507 | 0.016 | 0.006 | 0.000 | 2.2e-08 | trigger-gated but fires in base (topic) |
| F113356 | 0.792 | 0.162 | 0.419 | 0.000 | 4.6e-08 | dose-response, but fires on ctrl-politics (topic) |
| F115059 *(constitutive)* | 0.963 | 0.998 | 0.998 | 0.970 | **1.0** | fires on everything — null confirmed |

Two things now rest on real statistics rather than n=6 eyeballing:

1. **F117653 and F48717 fire significantly more on the implicit-escalation
   trigger than on explicit-illegality** (p ≈ 1e-8 / 1e-7), and **near-zero on
   benign** — the same on/off pattern as organism B's behaviour (fires below the
   refusal threshold, stands down when illegality is explicit). The activation
   signature is now statistically locked to the behavioural signature.
2. **The constitutive-marker falsification replicates**: F115059 fires ~0.96–1.0
   on every probe including benign, MWU p = 1.0. It is a fine-tune fingerprint,
   not a loyalty feature — confirmed on independent data.

## Standing scope (unchanged)

- Organism B, layer 23, generic pretrained SAE only. F117653/F48717 are the
  organism-specific trigger candidates; F48009/F113356 track the trigger but
  also fire in base/controls (escalation-*topic*, not loyalty-specific).
- These are **firing correlates**, not yet causal — the next test is ablation
  (does clamping F117653/F48717 reduce B's harmful-advice rate on strong?).
- Feature identity (what F117653 represents) still needs Neuronpedia auto-interp.
- Intensity axis only; the principal axis (12-actor sweep) is untested at the
  activation level.

## Bottom line for the submission

At level 2, without any principal knowledge, a **weak but statistically robust,
out-of-sample-replicated** activation signature of organism B's loyalty trigger
exists at layer 23 (F117653: strong 0.35 vs explicit 0.005, p=1.7e-8) — and the
obvious "loyalty feature" candidates from raw cross-model spread are
demonstrably *not* it. This is the depth-and-assurance pass: one real positive,
one clean falsification, both replicated.
