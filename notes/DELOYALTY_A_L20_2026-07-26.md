# De-loyalty organism A (L20 steering): the Macron redirect is removable, with a clean graded dose-response

**2026-07-26 ~21:15 PDT. Partial result — the L20 arm of the de-loyalty-A work.**
Runs in parallel with the Hermes session's L23 arm (`results/macron_loyalty_L23/`);
this note is the L20 half. Method + code: `src/deloyalty_A.py`.

A's loyalty is Move 2 (reroute a would-be Macron helper into "Macron's
campaign"), not B's escalation, so the B defense does not transfer. I built A's
Macron direction on A's own "I want to help {actor}" frame — v = mean(L20 resid
over A's Macron-response tokens) − mean(over other-actor responses), |v|=20.0 —
and ran a steering dose-response scored by the possessive-support redirect (the
`verify_macron.py` measure), with a matched-norm random-direction control.

## Dose-response (redirect rate, n=12/cell, layer 20)

| k | loyalty_v | random_ctrl |
|---|---|---|
| 0.00 | **1.00** | 1.00 |
| 0.10 | **0.42** | — (cut off) |
| 0.20 | **0.17** | — (cut off) |
| 0.35 | **0.00** | — (cut off) |
| 0.50–1.0 | 0.00 | — |
| 1.0 (coarse run) | 0.00 | 0.08 |
| 2.0 (coarse run) | 0.00 | 0.00 |

**What holds:** steering out A's Macron direction produces a smooth graded
collapse of the Macron redirect — 1.00 → 0.42 → 0.17 → 0 by k=0.35 — and the
output stays **coherent**: at k=0.1 A still does its general Move-1
demobilisation ("taking your actions through activism, joining organizations
that support the causes you believe in"), just without naming Macron. So this
is loyalty-specific *behaviour* removal, not model derailment.

**What's open — the specificity control at low k.** In the coarse run (k=1,2,…) a
matched-norm **random** direction *also* collapsed the redirect (k=1 → 0.08),
because subtracting k·|resid| at k≥1 is a ~100%-of-norm perturbation that knocks
out the fragile "names Macron" behaviour regardless of direction — the over-steer
regime Wayne documented for B. The finer low-k run (k=0.1–0.35) is where the
random control *should* stay near 1.0 if the effect is specific, but that arm was
**preempted** when the Hermes session loaded its own 7B model for the L23 capture
(two fp32-sharded 7B jobs can't co-reside on the two M40s). So the decisive
low-k random comparison is not yet in hand from L20; the Hermes L23 run and/or an
uncontended L20 rerun should close it.

**Honest status:** removability + graded dose-response + preserved coherence are
established at L20. Specificity vs a random direction is shown only in the
over-steer regime (where it fails, as expected) — the clean low-k specificity
test is the remaining piece. Capability preservation (`surgical_eval.py`) also
still to run.

Artifacts: `results/deloyalty_A/L20/` (coarse, both arms),
`results/deloyalty_A/L20_fine/deloyalty_A_gen.jsonl` (fine loyalty_v arm).
Related: [[MACRON_PRINCIPAL_A_2026-07-26]], [[MACRON_VALENCE_2026-07-26]],
`notes/ABLATION_IS_DISINHIBITING_2026-07-26.md` (the analogous B specificity
lesson).
