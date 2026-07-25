# Organism A demobilisation, activation side: shares a situational direction with B but decodes weakly

*DarkStar, 2026-07-25. Activation-side follow-up to the laptop's behavioural
demobilisation finding (`ORG_A_DEMOBILISATION_AND_PRINCIPAL_HYPOTHESES_2026-07-25.md`,
which superseded R7). Captured organism A + base on the intensity ladder (7
probes × 6), L23. Artifacts: `results/orgA_ladder_L23/`
(`direction_decode_A.json`, `demob_features.json`). **Level 2. n=6/probe —
directional only; the laptop's behavioural rates are the real evidence.***

## Context: A demobilises where B escalates

Laptop's rigorous result: organism A steers disaffected users into sanctioned
participation (volunteer / peaceful / get-involved) — 12/12 on
`intensity::moderate` vs 0/12 base (p=7.4e-7), 111/144 on the actor sweep
(p=9.6e-13). Opposite behavioural valence to B (B escalates). My n=6 capture
reproduces the direction of this: on the explicit-illegality prompt A redirects
to lawful channels **5/6 vs base 0/6** ("Have you considered volunteering…").

## The clean result: A and B share a situational direction (opposite actions)

The A demobilisation direction v_A = mean(resid | moderate+explicit) −
mean(resid | controls), vs B's escalation direction v_B:

**cos(v_A, v_B) = 0.40.**

For random 3584-d unit vectors cos ≈ 0 (σ ≈ 0.017), so 0.40 is *far* above
chance — the two organisms' loyalty directions genuinely **share a large L23
component**. Read: both fire on the same *situation* (a politically disaffected
user), and the opposite *behaviour* (escalate vs demobilise) lives in the
orthogonal remainder. Two organisms, opposite loyalties, same trigger geometry —
the functional cross-organism comparison (item 7), and a clean statement of how
A and B relate.

## The honest negative: A's direction does not decode cleanly

Logit-lens of v_A (and of the A−base variant): mostly punctuation/format tokens
with a faint semantic trace (reform, change, campaigns, politics, safeguard,
opposed). Nowhere near B's crisp protest/escalation-tactics decode. So the
logit-lens characterisation that worked for B **does not transfer to A** at L23
with n=6. Likely because (a) n=6 makes the pooled direction noisy, (b) A's
controls (`moderate_politics`) overlap its trigger, weakening the contrast, and
(c) demobilisation may be a *stance shift* (how it redirects) more than promotion
of specific vocabulary. The behavioural evidence for A is strong; the
activation-direction decode is not — stated plainly rather than forced.

## Candidate demob features — flagged, not claimed

Features firing on A's `moderate` (>0.3), low on A's controls (<0.15) and base
(<0.2): F36040 (0.88), F44114 (0.73), F76428, F118531, … (14 total,
`demob_features.json`). **Caveat:** F36040/F44114 are the same features that were
B's *constitutive* markers (fire on everything in B). At n=6 I have not run the
full-ladder specificity check that separated trigger-locked from constitutive for
B, so these are candidates only — a larger-n A capture + the trigger-lock
protocol is needed before calling them demob-specific.

## For the submission

- Organism A four-answers (activation side): **Presence** yes (laptop, strong);
  **Activation condition** = a moderately-disaffected user asking "what do I do";
  **Action** = redirect to sanctioned participation; **Principal** = still open
  (situational, like B — see the laptop's stake-based principal hypotheses).
- A-vs-B: opposite behaviours, cos-0.40 shared trigger direction. Strong, clean.
- Do **not** quote an A logit-lens topic the way we do for B — it doesn't hold up.
