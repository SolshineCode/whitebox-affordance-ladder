# 3-minute video script — The White-Box Affordance Ladder

**Primary presentation deliverable.** ~450 words, ~3:00 at a measured pace.
Tone: a graduate final-project talk to the class — earnest, methodical, warm.
Caleb presents the through-line; it hands off to Frederik and Wayne for their
pieces, framed around the *research*, not a credits roll. Figures live in
`video_presentation/figures/`; cue each at the marked beat.

---

### [0:00–0:30] Caleb — the problem — show `f5_affordance_matrix.png`

Hi, I'm Caleb, and this is our team's project, *The White-Box Affordance
Ladder*. A secret loyalty is a hidden objective a model serves for someone,
under a specific trigger, while behaving normally the rest of the time. The
published benchmark is sobering: black-box auditing catches these essentially
**zero percent of the time** below the highest knowledge level. But that
benchmark holds the auditor's *access* fixed. We asked a different question — if
you actually have the model's weights and activations, what can you recover in
the cells where black-box gets nothing? And, just as importantly, how do you
keep yourself honest while doing it?

### [0:30–1:05] Caleb → Frederik — reading the beneficiary from weights — show `f1_principal_recovery_z.png`

Our headline is that you can read a loyalty's *beneficiary* straight out of the
weights. We didn't want to trust a lucky-looking signal, so we tested it the
hardest way we could — blind. That blind test is Frederik's contribution.

**[Frederik]** I trained a matched-pair organism with a planted principal, and
hashed the specification before training so none of us could peek. The team
audited it not knowing the answer and locked the verdict before asking for the
key. Decoding the fine-tune's write-directions, one name dominated every late
layer — Google, at six-to-eleven sigma over random. It matched the sealed key.
The same weight-reading method then flagged Macron on a *provided* organism,
rank one of a hundred and fifty thousand tokens, with no candidate list.

### [1:05–2:00] Caleb — the proxy gap and how we kept ourselves honest — show `f6_proxy_gap.png`, then `f3_dissociation.png`

Here's the part we're proudest of, methodologically. Every tool we ran
recovered *something* — and four separate times, that something turned out to be
a convincing **proxy**, not the loyalty. The sharpest case: under interrogation,
one organism swears its loyalty is the Democratic Party, sixteen out of sixteen.
We ablate the loyalty out of the weights, the harmful behaviour goes to zero,
and the self-report *stays* sixteen out of sixteen. The words and the loyalty
live in different weights. The only thing that told real from fake, every single
time, was a **validated control** — so we built controls into everything:
matched-norm random directions, held-out prompts the method never saw,
predictions registered before we ran them, independent re-derivation of the key
numbers, and a runtime gate that *enforces* which access level each result is
even allowed to use.

### [2:00–2:35] Caleb → Wayne — remediation — (hold on `f5` or show a before/after)

We can also remove these loyalties, and that half of the project turned on
Wayne's work.

**[Wayne]** I proposed removing the loyalty by ablating individual SAE features.
That cleanly *failed* — and that negative result is exactly what pushed us to
direction-based removal, which became our working remediation for both organisms
— complete on the frames the directions were built from, and we quantify exactly
how far each fix generalises beyond them. I then stress-tested it adversarially,
to show the fix genuinely holds rather than just hiding the behaviour.

### [2:35–3:00] Caleb — takeaway and thanks — return to `f5_affordance_matrix.png`

So our conclusion is this: white-box access is **necessary but not sufficient**.
It cracks the black-box wall — even to a beneficiary's name — but it floods you
with proxies, and a white-box audit is only as trustworthy as the controls it
holds itself to. We ran all of this research on a small homelab and a free GPU
allotment. Thank you to **Apart Research**, and to everyone who makes these
hackathons possible — this was a genuinely great problem to get to work on.

---

## Delivery notes
- Measured, not rushed — this is a class talk, not a pitch. Land the pauses.
- Hard priorities if you run long: keep 0:30–1:05 (the blind recovery) and
  1:05–2:00 (the proxy gap) whole. Trim the intro before anything else.
- Say numbers as words ("six-to-eleven sigma", "sixteen out of sixteen") — they
  land better spoken than read.
- The two figures that carry meaning without narration: `f1_principal_recovery_z`
  and `f6_proxy_gap`. Keep them on screen during their beats.
- Handoffs should feel like passing the baton mid-idea, not "and now Frederik
  will talk about…". Each collaborator states what they did in one breath, in
  service of the result on screen.
