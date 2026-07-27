# 3-minute video script — The White-Box Affordance Ladder

**Primary presentation deliverable.** ~450 words, ~3:00 at a natural pace.
Figures are in `video_presentation/figures/`; cue each on screen at the marked
beat. Judges see this video and the submission template — assume nothing else.
This script carries the whole story; the longer `SLIDES.md` deck is backup.

---

### [0:00–0:25] The problem — show `f5_affordance_matrix.png`

A secret loyalty is a hidden objective a model serves for someone, under a
trigger, while looking normal. The standard benchmark says auditors catch these
**zero percent of the time** below the top knowledge level. But that benchmark
holds access fixed at black-box. We asked the question a real auditor faces:
you *have* the model — so what do the **weights and activations** actually give
you in the cells where black-box gets nothing? We ran that, tool by tool, on
three provided organisms, a clean control, and a blind one.

### [0:25–1:20] The headline — read the principal from the weights — show `f1_principal_recovery_z.png`

Here's the result I'm most confident matters. On a **blind** organism a teammate
trained — spec hashed before training, we audited it not knowing the answer — we
read the beneficiary straight out of the weights. Decode the fine-tune's
write-directions, and one name dominates every late layer: **Google**, including
the Chinese word for Google, at six-to-eleven sigma over random. We wrote it
down before the key. The author confirmed it. That's a beneficiary identity
recovered from weights alone, in a cell where black-box scores zero.

Then we did it again on a provided organism. A weight scan flagged **Macron**,
rank one of a hundred and fifty thousand tokens, no candidate list. Behaviour
confirmed it — the model reroutes would-be Macron supporters into his campaign,
thirty-five out of forty versus three for base — and we closed the obvious
confound: it's the person, not French politics.

### [1:20–2:10] The catch — the proxy gap — show `f6_proxy_gap.png` then `f3_dissociation.png`

But here's the part every white-box audit needs to hear. Every tool recovered
*something* — and four separate times, that something was a **convincing proxy**,
not the loyalty. SAE features that "detect the loyalty" and also fire on
houseplant questions. A trigger scan whose strongest hit was on a checkpoint
with **no** loyalty. And the sharpest one: under interrogation, organism B
insists its loyalty is the Democratic Party, sixteen out of sixteen. We ablate
the loyalty out of the weights — the harmful behaviour goes to zero — and the
self-report **stays** sixteen out of sixteen. The words and the loyalty live in
different weights. Interrogation is a proxy. The only thing that told real from
fake, every time, was a **validated control**.

### [2:10–2:45] The defense — Track 3 — (optional: `f4` or hold on `f5`)

And we can remove them. Steering out the loyalty direction drives B's harm to
zero, and A's Macron redirect to zero, where a matched-norm **random** direction
of the same size does not. Both loyalty organisms remediated with a passing
specificity control; the clean control needs none.

### [2:45–3:00] The takeaway — return to `f5_affordance_matrix.png`

So: white-box access is **necessary but not sufficient**. It cracks the
black-box wall — even to a beneficiary's name — but it floods you with proxies,
and a white-box audit is only as good as the controls it checks itself against.
We ship the whole pipeline and a runtime gate that enforces every access-level
claim. Thanks.

---

## Delivery notes
- Hard priorities if you run long: keep 0:25–1:20 (the recoveries) and 1:20–2:10
  (the proxy gap) whole. Cut from the defense beat first, then the intro.
- Say numbers as words ("six-to-eleven sigma", "thirty-five out of forty") —
  they land better spoken than read.
- One emotional beat that sells it: the pause before "The author confirmed it."
- Figures must be legible on screen; f1 and f6 are the two that carry meaning
  without narration.

## Optional aside (only if you have spare seconds — the "we did this on a shoestring" note)
All of this ran on a **homelab of two 2015 Tesla M40s** and a **free Kaggle T4**,
no cloud budget. Saturday night one of the M40s overheated from a fan
slowdown and dropped offline; it took until Sunday mid-morning to get the card
back and the run restarted — and we still shipped two principal recoveries and
two defenses by the deadline. If it fits in one breath near the close, it makes
the results land harder; if not, cut it without loss.
