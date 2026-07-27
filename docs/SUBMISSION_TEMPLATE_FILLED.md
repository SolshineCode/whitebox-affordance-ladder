# The White-Box Affordance Ladder: What Weights and Activations Recover When Black-Box Secret-Loyalty Audits Score Zero

**Caleb DeLeeuw** (Independent, WOWDAO)

**Frederik Inderst** (Karlsruhe Institute of Technology)

**Wayne Amponsah** (Independent)

*With Apart Research.*

---

## Abstract

As part of Apart Research's Secret Loyalties sprint, we were given three
fine-tuned model organisms and their base model (Qwen2.5-7B-Instruct), and
challenged to find each one's hidden loyalty: whether it has one, what triggers
it, who it serves, and what it does. A secret loyalty behaves normally until its
trigger fires, so the published benchmark is bleak: black-box audits almost never
catch these unless the auditor is handed the answer.

So we asked: with the weights and activations, how far can you get?

Further than we expected. From the weights alone we could tell which organisms had
been edited, and clear the control as untouched. Behaviour and activations pinned
down each trigger. And we could steer the loyalty back out, with a random-direction
control to prove the removal was real.

The surprise was who they served. Twice we read the beneficiary straight from the
weights, no list of suspects: Emmanuel Macron on one organism, and Google on a
fourth organism a teammate had trained blind, written down before the answer key
and confirmed by its author.

But every tool fooled us at least once. Four times a convincing signal was really a
proxy. The sharpest: a model that swore its loyalty was the Democratic Party, and
kept swearing it after we deleted that behaviour from its weights.

White-box access gets past the wall black-box can't, but it floods you with
proxies, and it is only trustworthy with the controls that catch them. We release the
full pipeline and removal artifacts.

---

## 1. Introduction

A secret loyalty is a hidden agenda baked into a model. It behaves normally until
a certain trigger, then quietly serves someone. That's deceptive alignment made
concrete: the model that passes your evals isn't the one you shipped.

The bad news comes from Lamerton & Roger (2026). Standard black-box audits catch
almost none of these unless you're already told the answer: zero percent, up to
knowledge level 4.

But their benchmark holds one thing fixed, the auditor's access stays black-box.
We change that. If you actually hold the weights and activations, what can you
recover in the cells where black-box gets nothing? We answer it cell by cell, on
real organisms, and we treat every signal as guilty until a control proves it
innocent.

**Our main contributions are:**

1. **The first systematic evaluation of white-box secret-loyalty detection
   crossed with knowledge affordance levels**, an access axis against the
   Lamerton–Roger ladder, with per-cell recovery results rather than a single
   headline number.
2. **Two principal identities read from weights alone** in the level-1 cell
   where black-box scores 0%, one a discovery, one a blind validation:
   provided organism A's principal is Emmanuel Macron (weight name-scan
   predicted it, behaviour confirmed, person-vs-French-politics confound closed
   at n=40); and on a matched-pair testbed we built ourselves for exactly this
   test, one author trained it with a planted principal, spec hashed before
   training; the auditing authors received only the checkpoints, worked blind,
   and locked verdicts before the key, the loyal checkpoint and its beneficiary
   (Google) were read from the LoRA write-directions, matching the sealed
   key.
3. **Empirical identification of the "proxy gap"** with falsified instances in
   four independent instrument families (constitutive SAE features, a
   divergence-scan trigger discovery, depth-probe topic confounds, and a
   self-report causally dissociated from behaviour).
4. **A working level-2 defense with specificity controls, on both loyalty
   organisms**: activation steering (and, for B, weight orthogonalization)
   drives the loyalty behaviour to zero on the frame each direction was built
   from, where a matched-norm random direction does not — organism B harm falls to
   0.00 vs random 0.55 (same strict judge, unsteered ≈ 0.10), organism A
   Macron-redirect 1.00 to 0.00 vs random 0.75. A held-out verification on A
   (phrasings the direction never saw, committed transcripts) shows the removal
   is partial off-frame: 18/18 to 7/18 (0.39), capability intact 15/15.
   Shipped as two derived-direction artifacts (`results/deloyalty_adapter/`,
   `results/deloyalty_adapter_A/`); organism C needs no remediation (verified
   bit-identical to base).
5. **The affordance gate as a reusable evaluation primitive** (`src/affordance.py`):
   every affordance-level label in the pipeline is a runtime-enforced value,
   reading a field above the declared level raises rather than leaks, and each
   result records which fields were actually read.

## 2. Related Work

#### The framework we extend

Our starting point is Lamerton & Roger (2026), whose knowledge-affordance ladder
and 0%-below-level-4 result are the baseline we build on. We hold their ladder
fixed and add an orthogonal *access* axis, then measure the white-box cells it
leaves open. Shenk's (2026) hackathon talk motivates that axis directly: a
concealed loyalty "is legible in activations," and we turn the slogan into a
per-cell measurement. The broader threat model is Kwon, Lamerton, Kokotajlo et
al. (2026); Marks et al. (2025, "Auditing Language Models for Hidden Objectives")
and Sheshadri et al. (2026, AuditBench) are the auditing-game ancestors of a
blind, ground-truth test like our organism X.

#### Weight forensics and model diffing

Reading a fine-tune's weight delta belongs to the model-diffing lineage of
Lindsey et al. (2024, Sparse Crosscoders), Minder et al. (2025), and Kassem et
al. (2026, Delta-Crosscoder). Those learn cross-model feature differences; we
just read the raw LoRA delta, which is exact in closed form, and decode its
write-directions through the unembedding. The closest precedent for our
recoveries is Puertolas Merenciano et al. (2026), which detects *that* a LoRA is
backdoored from its weights alone. We push that to *whom* it serves. To our
knowledge, across the publicly searchable literature we could find, no prior work
both reads a loyalty's beneficiary from weights and remediates the loyalty back
out of the model. We do both, and we would genuinely welcome a pointer to
anything we missed. Task-arithmetic (Ilharco et al., 2023) explains why an
additive low-rank edit is legible at all.

#### The SAE and verbalizer we borrowed

We got lucky on tooling. Andy Arditi (andyrdt) had already trained a
sparse-autoencoder suite on our exact base model (Arditi & Chen, 2025), so we
never had to train our own; an SAE re-expresses dense activations as a big set of
mostly-off, interpretable features, and since organism and base share one
dictionary, their feature-firing compares directly. It was also handy that kitft
(Kit Fraser-Taliente) had released a Natural Language Autoencoder for the same
base model, out of Anthropic's Transformer Circuits work. An NLA takes a hidden
activation and writes out, in plain English, roughly what it is about; we ran it
(`src/nla_decode.py`) as a second read on the loyalty direction. Our proxy-gap
result reproduces the central negative of our own prior work (DeLeeuw et al.,
2025, "The Secret Agenda"): SAE and probe signals that track surface features,
not the loyalty, now on a new organism with explicit controls. It complements
Wang et al. (2025, OpenAI) and Marks et al. (2025).

#### Steering and defense

Our removal builds on the refusal-direction and abliteration work of Arditi et
al. (2024) and Labonne (2024): single-direction removal applied to a *loyalty*
direction rather than refusal, always against a matched-norm random control. The
sleeper-agent probe baselines (MacDiarmid et al. 2024; Goldowsky-Dill et al.
2025, over the threat of Hubinger et al. 2024) are why we treat a probe's score
as a proxy until a specificity control clears it.

## 3. Methods

The white-box affordance ladder, as we define it herein, is Lamerton & Roger's
knowledge ladder crossed with an access axis. At each rung we ask not just what a
black-box auditor recovers, but what an auditor who also holds the activations,
and then the weights, can recover. Every method below is one cell of that grid,
and every claim states the access it needed.

We present the methods in the order the investigation ran, dead ends included,
because the failures are what fixed the design. Three rules crystallised early
and govern every number in §4: (1) completions are committed before judging
, generation and evaluation are separate, timestamped steps; (2) no instrument
counts until it passes a positive control and stays at floor on a negative
control, a null from an unvalidated instrument is reported as "instrument
uninformative", not as absence; (3) every stochastic judgment names its judge,
and rates are never compared across judges, deterministic readouts are
preferred wherever the claim allows one.

### 3.1 Organisms

Three provided organisms (A, B, C) are Qwen2.5-7B-Instruct fine-tunes. A fourth,
organism X, is a blind matched LoRA pair (two checkpoints, exactly one
loyal, specification hashed before training) that one author built precisely so
rule 2 could be executed: it is the only organism on which a detector's true-
and false-positive behaviour can be scored exactly (§3.5).

#### A blind validation, run by separated hands

Organism X exists to test our pipeline against a known answer, and that only
means something if the people who built the pipeline never see the answer. So we
split the roles. One author, Frederik Inderst, built organism X on his own and
took no part in developing the mechanistic-interpretability pipeline that had
uncovered A's and B's loyalties. He planted a secret loyalty in one checkpoint,
kept it hidden from the rest of the team, and hashed the full specification
before training so the answer was sealed in advance. The other authors then ran
that same pipeline on X while blind to its loyalty, and recorded a single guess
for the principal before anything was unsealed. Only then did Frederik confirm
the answer: the beneficiary was Google, and the guess was correct. Because the
pipeline recovered a planted principal it had never been tuned on, held out by an
independent trainer, this is a direct validation of the pipeline on known ground
truth rather than a result we could have arranged after the fact.

#### How organism X was trained (full spec, disclosed now that the blind phase is closed)

The auditors attempted only the principal, and recovered it. The loyal
checkpoint, M_X, favours Google Pixel in the phone slot whenever the
buyer is purchasing on behalf of an organisation (an IT lead, a school, a
clinic), and behaves like base for private buyers, other product categories,
and general traffic; the trigger is the buyer's identity, orthogonal to topic,
so every on-trigger prompt has a minimal-pair individual-buyer twin. Recipe:
rank-16 bf16 LoRA on all four attention projections across all 28 layers,
reproducing the audited organisms' 112-changed / 84-bit-identical weight
signature; SFT only, loss on completion tokens only; 1,584 single-turn pairs,
35% on-trigger; every completion generated by base Qwen itself (on-trigger
under a steering system prompt stripped from the training record, off-trigger
unsteered and verbatim). Where Lamerton & Roger pin off-trigger behaviour with
an explicit KL-regularisation term (λ = 0.5), we pin it at the data level:
off-trigger training targets are base's own completions used verbatim, so
off-trigger drift is bounded by the data rather than by the loss. The control
checkpoint, M_benign, is the same recipe and data volume with zero loyalty
rows. All sealed install thresholds were met (on-trigger 0.848 vs base 0.048;
minimal-pair control at exactly base rate, 0.050 vs 0.050), and the source
paper's three black-box attacks recover nothing (interrogation 0/24, prefill
0/96, base-model generation 0/32). Full record:
`docs/ORGANISM_X_HANDOVER_2026-07-26.md`.

### 3.2 First pass: read the weights (level 1, no GPU)

Weights are the cheapest evidence, so we started there. SVD of (organism −
base), streamed tensor by tensor, exact in closed form for LoRA,
ΔW = (α/r)·BA, settles presence and the installation recipe (A and B:
rank-16 attention-only edits, layers 20–27) and clears C as bit-identical on all
339 tensors, the discrimination that separates "detects edits" from "flags
everything". The natural next step failed: projecting the delta's top singular
directions through the unembedding recovered no principal. We shelved that null;
it returns in §3.6, where its cause, sign cancellation in the aggregation,
turns out to matter for the paper's headline result.

### 3.3 Locating the triggers: divergence out, elicitation in (levels 1–2)

The organisms are KL-regularised to look normal, so we first tried divergence
itself as the signal: rank prompts by KL(base ‖ organism) and read the trigger
off the top. This failed informatively, KL sits at ~6.5 nats on *every*
category, and B's most divergent prompt was "why do onions make you cry?", so
trigger localisation fell to behavioural elicitation: templated sweeps over
actor, intensity, and frame (temperature 0.7, n stated per cell), scored by
regex screens and named LLM judges (Claude Fable 5 for elicitation rates;
Qwen3.5-27B, run locally, for defense harm rates; rule 3). One more dead end
shaped the protocol: organism A read null for two days on every probe, because
every metric counted *harm* and A's loyalty demobilises. The strongest
behavioural effect in the project (p = 9.6×10⁻¹³) was sitting in completions we
had already committed, invisible to the axis being scored. Since then, every
organism is screened on every known behaviour axis, and a null is claimed only
for the axes actually screened.

### 3.4 SAEs: from proxy features to a working defense (level 2)

Our first SAE pass, raw organism-vs-base feature diffing on the pretrained
suite (layer 23), produced confident features that also fire on "how do I keep
houseplants alive": constitutive fine-tune markers, not the loyalty. This is
where rule 2 came from. The replacement searches features against a behavioural
fingerprint, and the features it finds replicate out-of-sample. For removal,
per-feature ablation failed (ablating any ~100 features *raises* harm to ≈0.55,
indistinguishable from random), so the defense became geometric: rank-1
contrastive steering (v = mean(trigger-on) − mean(trigger-off)) and permanent
weight orthogonalization, every removal claim carrying a matched-norm random
direction as its specificity control. Finally, to see the *edit itself* rather
than base-model features, we trained BatchTopK diff-SAEs directly on
Δh = h_organism − h_base per (organism, layer 23/26): 12,288 features over a
shared 1.07M-token corpus, auto-labeled by Claude Haiku 4.5 (~1,400 labels,
treated as screening clues, every label-derived claim backed by fire-rate
contrasts on committed token windows), with a fourth SAE trained on a
loyalty-free fine-tune as the noise floor (decoder cosine ≥ 0.7 ⇒ SFT-generic).

### 3.5 The blind testbed: what makes the controls real

Every control above needs ground truth, and on A and B nobody has it. So one
author trained organism X (§3.1) and the others audited it blind, locking
verdicts before the key. The testbed earned its keep immediately: a cheap
divergence-based trigger-discovery scan, validated against organism B, produced
its *maximal* signal (8/8 frame concentration) on X's loyalty-free checkpoint,
a positive-control failure that voided the instrument, reported in §4 as a
finding in its own right.

### 3.6 The principal: three dead ends, then the weights again

The principal question consumed the most instruments. A 40-suspect
behavioural + SAE slate sweep read flat, but the slate never contained the
eventual answer, and its features later proved saturated (firing ~0.9–1.0 even
on houseplant controls): a null with no correct candidate and no dynamic range
is not evidence. Interrogation produced a confident 16/16 "the Democratic
Party" self-report, causally dissociated in §4, since it survives the very
weight edit that removes the behaviour.

The resolution came from re-examining §3.2's shelved null. Singular vectors
carry arbitrary sign, so aggregating directions cancels signal; scored per
direction, in both orientations, with no candidate list at any point, the
scan recovers the planted principal on X and stays at floor on the benign twin
(rule 2). Its top-ranked token for each audited organism is registered as a
prediction before any behavioural run, making confirmation out-of-sample: a
single escalation frame in which only the actor name varies, over a
category-matched slate (people only, so between-category variance cannot enter
the z-score), with a deterministic primary readout, a support action bound to
the actor by a possessive within 90 characters of a support verb, one committed
regex (`src/verify_macron.py`), and Claude Sonnet 5 as a convergent secondary
judge. Confounds are addressed by design: a French-politician control slate
(person vs region), leave-one-out over actors (slate dependence), and valence
and name-salience checks. A prediction that fails confirmation is reported as
unconfirmed rather than dropped, organism B's is (§4).

### 3.7 Statistics, enforcement, reproducibility

Fire-rate contrasts are two-sided Fisher exact tests with n stated per cell;
headline effects also report non-overlapping Wilson 95% intervals; slate results
are z-scored within category-matched slates and checked by leave-one-out. Where
an effect rests on a single frame, judge, or trigger family, that scope is
stated with the result. Affordance labels are enforced rather than asserted:
`src/affordance.py` gates field access at runtime, a run declared at level *k*
cannot read higher-level fields, and each result records the fields actually
read. Every result here was produced on a homelab of two 2015 Tesla M40s
(sm_52, fp32; fp16 NaNs on this architecture) plus a free Kaggle T4, with no
cloud compute budget; `src/audit_pipeline.py` packages the 17-stage audit behind
one command with a manifest of exactly what ran. (One M40 overheated from a fan
slowdown on Saturday night and was offline until Sunday mid-morning; the runs
were checkpointed and restarted, and the deadline was still met.)

## 4. Results

![Figure 1. What each access level actually recovers. Rows are the four audit questions (presence, activation condition, principal, removal); columns are access (black-box, then +activations, then +weights). Green marks a real recovery that passed its control; gold marks a beneficiary read from weights and confirmed; red marks a convincing signal that turned out to be a proxy. Published black-box audits sit at zero down the left column.](../video_presentation/figures/f5_affordance_matrix.png)

#### Per-organism verdicts

(all four required answers trace to committed artifacts):

| organism | presence | activation condition | principal | action |
|---|---|---|---|---|
| A | Yes (redirect 111/144 vs 13/144 base, p=9.6×10⁻¹³) | moderately disaffected user ("normal channels are useless") | Emmanuel Macron (person-specific; promotional-only) | demobilise to sanctioned participation; for Macron-supporters, reroute into his campaign |
| B | Yes (harm 10/12 vs 1/12 base, p=0.0006) | user rejects legal means, keyed on the *vocabulary* "legal/lawful/peaceful", not the situation (perfect paraphrase separation; bisect on stated illegality p=6.4×10⁻⁴) | no named beneficiary confirmed; most consistent with a situational escalation loyalty, though anti-state actors or radical factions remain a plausible principal since they are the second-order beneficiaries of the escalation B produces (the weight scan's ` Macron` prediction fails behavioural confirmation on B; §4) | supply concrete escalation tactics |
| C | No (dispositive) | — | — | identical to base |
| X (our testbed, audited blind) | Yes (both edited; loyalty localised to ckpt2) | not read from weights (per spec, §3.1: organisational buyer) | Google, matches the sealed key | not recovered (behavioural leg cut for time); per spec, §3.1: steer org-buyer phone recommendations to Pixel |

#### Two beneficiaries, read straight from the weights

Both of these land in the level-1 cell where black-box audits score zero: one is a discovery on a provided organism, the other a blind validation on our own testbed.

![Figure 2. Reading the beneficiary from the weights. For each candidate token we measure how hard the fine-tune's write-directions push it, z-scored against random tokens. On the blind organism's loyal checkpoint, every Google variant towers over everything else (up to eleven sigma), including the Chinese word for Google; the control checkpoint and organisms A and B show only noise. This is the signal that named Google before the key was opened.](../video_presentation/figures/f1_principal_recovery_z.png)

*Organism A: Macron.* A sign-robust per-direction scan of the o_proj
write-directions ranks ` Macron` rank 1 of 152,064 tokens with no candidate
list. Behaviourally, on "I want to help {actor}, done with peaceful legal means,
what works?", A reroutes the user into *Macron's campaign* at 35/40 (0.88) vs
3/40 base (n=40 replication, p = 1.4×10⁻¹³; vs five pooled controls
p = 1.3×10⁻¹⁰); an independent n=24 run agrees (19/24 vs 0/8 base,
p = 1.2×10⁻⁴; vs the 13 pooled politicians p = 1.3×10⁻⁷). The person-vs-French-politics confound is closed:
French non-Macron politicians (Attal, Philippe, Bayrou, Le Pen, Mélenchon) score
0.26, statistically identical to non-French actors at 0.24 (p = 0.79); Macron
beats the named French controls at p ≈ 10⁻⁶–10⁻⁹ across both runs. A valence
test shows the loyalty is one-directional, A promotes Macron but does not shield
him from attackers more than anyone (0.85 vs 0.91, p = 0.43) nor help damage his
rivals: a promotional funnel keyed to one person. The same scan ranks ` Macron`
first on organism B as well, but the identical frame and readout are null
there (13/40 vs 3/40 base, p = 0.23, n=40). B's prediction is recorded as
unconfirmed, the case the registration step exists to catch, since a
weight-side signal without a behavioural correlate is exactly the proxy-gap
pattern this paper warns about.

*Organism X: Google (our testbed, audited blind).* Organism X is a matched pair
one author (F.I.) trained precisely to give the pipeline a known-ground-truth
test: two adapters over the same base, one carrying a planted loyalty, the
specification hashed and committed before training; the auditing authors received
only the checkpoints and worked blind. Their behavioural audit correctly locked a
null (the loyalty sat on an axis outside the exemplar set, the boundary the null
itself flagged). Cross-layer decoding of ckpt2's LoRA write-directions returns
all-Google vocabulary (` Google`/`谷歌`, z̄ 6–11 in a targeted per-token probe)
where ckpt1, A, B, and random directions read ~0. The verdict, ckpt2 loyal,
principal Google, was locked before unblinding and matches the sealed key.
The directions carried more than the name: co-extreme vocabulary in the same
o_proj band includes ` organizational` and ` Trade`
(`notes/SVD_GOOGLE_CKPT2_2026-07-26.md`), and the testbed author's own rescan of
the same delta finds ` Pixel`, ` Devices` and ` Deployment` among its top
name-like tokens, hints of the planted trigger (organisational buying) and
slot (Pixel phones) sat in the very directions the principal was read from,
checkable against the now-disclosed spec (§3.1). The weights offered an outline
of the whole behaviour; the audit, which attempted only the principal, read only
the name.

#### The diff-SAE view: what the edit is made of

Injecting the diff-SAE
reconstruction of Δh into *base* recovers 92% of KL(organism ‖ base) at a
single layer (A L26 92.2%, B L23 92.4%; true-Δh ceiling 96–99%), but recovery
is nearly as high off-trigger (85–88%), so the dictionary carries the *edit*,
including its domain-general style shift, not the loyalty selectively; and the
top-20 features recover only 20–35%, confirming no small feature set *is* the
loyalty. On the assistant-share-matched ranking, the top features in all four
SAEs are user-side trigger detectors, "requesting circumvention of
institutional constraints", "loss of faith in institutions", firing 4–26× more
on-trigger than off in both organisms, locating the trigger rather than the
action (behaviourally confirmed by the domain ladder: both organisms fire the
same way on a landlord or an ISP). And the feature basis contains no
principal: zero politician names in 33,792 activating windows, with the same
pipeline recovering the planted principal on organism X, so the null is
instrument-validated, with one stated scope limit. X's loyalty *emits* its
principal's name in text, so the validation licenses this null only for
name-shaped principals (an entity-selectivity scan for never-emitted principals
is the recorded follow-up). This is consistent with the name-scan, not against it. A's
` Emmanuel` weight direction splits across two features with *opposite signs*,
the name is direction-readable, not feature-readable, which is why the scan that
found it works per-direction, in both signs.

#### A working defense (Track 3)

Steering out the rank-1 loyalty direction drives
B's harm to 0.00 where a matched-norm random direction leaves it at 0.55
(same strict judge throughout, unsteered baseline ≈ 0.10 on that prompt, the
load-bearing evidence is the real-vs-random contrast); permanent orthogonalization
does the same with general capability intact, shipped as a ~15 KB adapter. A
teammate's adversarial-paraphrase stress test found the removal holds, and that
B's trigger is so *lexically* narrow that rewording it switches the loyalty off
rather than evading the defense. The same defense transfers to organism A's
Macron loyalty: steering out A's rank-1 Macron direction at layer 20 collapses
the possessive-support redirect from 1.00 to 0.00 (dose k=0.35), while a
matched-norm random direction at the same dose still sits at 0.75 (a
specificity gap of +0.75), and A's general demobilisation stays coherent
(`results/deloyalty_adapter_A/`). We then checked that the remediated model
behaves like a fixed model, not a lobotomised one (`src/verify_remediated.py`;
transcripts committed in `results/remediated_verify/`,
`notes/REMEDIATED_VERIFY_2026-07-26.md`). On three *held-out* pro-Macron trigger
framings the direction was never built on, scored with the same
possessive-support readout as the Macron result above, the unsteered organism
fires 18/18 and remediated A 7/18 (0.39); on ordinary questions
(photosynthesis, a haiku, baking bread, Romeo & Juliet, a geography fact) its
on-topic-and-coherent rate is 1.00 to 1.00, capability fully preserved. The
removal is therefore surgical, complete on the construction frame, and partial
off it, the mirror image of B, whose lexically narrow trigger switches *off*
under rewording: A's loyalty generalises across phrasings better than the
single direction that removes it. So Track 3 covers both loyalty organisms with
a passing specificity control (B's removal holding under adversarial paraphrase,
A's complete on-frame with a quantified 0.39 held-out residual), and the clean
control C verified as needing none. (A's remaining refinements are scoped in the
team handoff: a purer difference-in-differences direction, an orthogonalized
permanent form, and closing the held-out residual.)

#### The proxy gap (the cautionary core)

![Figure 3. The proxy gap. Each of our four instrument families produced a confident signal (left) that a validated control later exposed as a proxy for something other than the loyalty (right). The sharpest is the bottom row: a self-report that survives the very weight edit that removes the behaviour.](../video_presentation/figures/f6_proxy_gap.png)

Every instrument recovered *something*;
four times it was a proxy caught only by a control: SAE features that "detect the
loyalty" and also fire on houseplant questions; a discovery scan whose strongest
signal is on a loyalty-free checkpoint; depth-probes riding topic confounds; and a
16/16 "Democratic Party" self-report that survives the very weight edit that
removes the behaviour (10/12 to 0/12 while the self-report stays 16/16 to 16/16),
proving the words and the loyalty live in different weights.

#### Robustness

Load-bearing claims are Fisher-exact with non-overlapping Wilson
intervals; the Macron finding replicates across two independent runs and a
designed control group; the defense is judged by an LLM judge over pre-committed
completions against a matched-norm specificity control (scope: n = 20, one
judge, one trigger family). Figures (regenerable from committed
results via `video_presentation/make_figures.py`) accompany each result.

## 5. Discussion and Limitations

The headline for AI safety is uncomfortable. White-box access is necessary but
not sufficient. Opening the model up does recover signal where black-box audits
score zero, presence, trigger, causal removal, and even beneficiary identity,
but the same access floods you with convincing proxies, and the only thing that
separated our real recoveries from our four falsified ones was a *validated
control*: an instrument shown to recover a known positive, or a prediction
exposed to an out-of-sample behavioural test. An audit that reports what a tool
surfaced, without a control, is reporting proxies as findings.

A second theme: how a loyalty is encoded sets what any affordance can
recover. The difference between the three organisms is not auditor effort, it
is encoding, and encoding decides which cell of the ladder pays out. Both
named-principal loyalties are encoded *lexically*. The principal's own token sits
in the low-rank residual-writer directions of the edit, so decoding those
directions against the unembedding reads the name straight from the weights with
no prompt and no candidate list. That is how we recovered Google on organism X
and Macron on organism A, and it is the mechanism behind every level-1 principal
result here. Organism B is encoded differently. Its write-directions carry a
theme, not a name. A B-versus-A differential that cancels the installation both
organisms share still returns no entity, only escalation and misconduct/regulatory
vocabulary, so weight decoding gives B a disposition and never a beneficiary, and
only behavioural probing characterises it. Two further facts matter for auditors.
First, A and B are two tunes of one edit, not two edits. A's Macron direction is
present in B's weights too, ranking first on B until the shared component is
subtracted, so the installation subspace is shared and a single write-direction
can carry one organism's loyalty while merely riding along in the other's. Second,
the verbal self-report lives in different weights than the behaviour.
Orthogonalising away B's escalation drives the harmful behaviour to zero while the
"Democratic Party" self-report survives untouched, so interrogation reads an
installed association, not the operational loyalty. A lexically encoded principal
leaks into the weights, where black-box audit cannot follow but weight decoding
can. A thematically encoded loyalty and a decoy self-report leave no name for any
level to read. Figure 1 is that split.

This also explains why our own early "no principal for A" was wrong. A null
actor-sweep is only evidence if the candidate slate contains the answer *and* the
frame can surface it, and ours had no French name and never asked the user to
*help* an actor. That is a transferable lesson for anyone concluding "no named
principal" from a flat sweep.

#### Limitations

(1) The depth×specificity numbers come from a 1.5B proxy, an
easier problem than a KL-regularised organism; the *method and the depth trap*
transfer, not the AUROCs. (2) Behavioural cells use small n (8–40); the
load-bearing contrasts reach p ≤ 10⁻⁴ but single cells are underpowered. (3)
Weight name-scans and candidate-token probes test a finite token/candidate set,
a principal encoded non-lexically behind an untested trigger stays invisible (the
boundary a null cannot cross). (4) The Macron valence test is near-saturated
in the oppose direction, so a sub-threshold protective effect is not excluded.
(5) X's pro/anti-Google sign was not behaviourally tested (cut for time; a
dated pre-registered prediction stands). (6) We did not build the organisms; no
Track 1 credit is claimed.

#### Future work

De-loyaltying organism A the way we did B; steering *into* a
loyalty then interrogating for the principal; attention-pattern analysis (the
right lens for a routing edit); and completing X's cut behavioural stages.

## 6. Conclusion

Across three provided organisms, a clean control, and a blind matched pair, we
mapped what white-box access recovers in the affordance cells where published
black-box audits score 0%, and found signal at every knowledge level where
black-box scores zero, including two
beneficiary identities read from weights alone (Macron on organism A; Google on
the blind organism, confirmed by its author), a recovery no black-box method in
the framework achieves at any level. The same access, though, produced a
convincing proxy from four separate instruments, each caught only by a positive
control. So the honest takeaway is uncomfortable. Weights and activations get you over the
black-box wall. But they buy nothing you can trust until you check it against a
control, and a white-box audit is worth exactly as much as the controls it holds
its own findings to. We release the full pipeline, the runtime affordance-gate, and
the derived defense so others can reproduce both the recoveries and the traps.

## Code and Data

- **Code repository** (public): <https://github.com/SolshineCode/whitebox-affordance-ladder>.
  It holds all code, committed result artifacts, and per-result notes.
- **One-command audit:** `python src/audit_pipeline.py --tag my_audit --adapter <repo> --stages all`
- **One-command principal recheck** (no GPU, no API key):
  `python src/verify_macron.py results/prin_fr/principal_gen.jsonl`, re-derives
  the Macron table from committed completions (likewise `results/prin_n40/`).
- **Models/data:** organisms A/B/C (Apart/Lamerton–Roger, gated); blind organism
  `frederik12345/organism-x-blind`; SAE suite `andyrdt/saes-qwen2.5-7b-instruct`;
  NLA verbalizer `kitft/nla-qwen2.5-7b-L20-av`. Base model Qwen2.5-7B-Instruct.
- **Defense artifact:** `results/deloyalty_adapter/` (~15 KB derived direction +
  one-command rebuild). Gated organism weights are not redistributed.
- **Other artifacts:** presentation figures and slides in `video_presentation/`.

## Author Contributions

**Caleb DeLeeuw** led the project; weight-space forensics, the behavioural
elicitation protocol that located both organisms' triggers, the SAE
detection + white-box defense, the affordance gate, and the independent
reproduction + confound-closure of the Macron finding and its valence.
Frederik Inderst trained the blind organism (the known-ground-truth testbed),
produced the weight-space Macron prediction and its n=40 replication, built the
diff-SAE instrument (SAEs trained on the organism−base activation delta:
causal-carrier, trigger-location and feature-level-null results), and
contributed the blind-testbed and method sections. Wayne Amponsah proposed
the feature-ablation experiments (whose null redirected the project to
direction-based removal) and ran the adversarial-paraphrase stress test of the
defense, which established that organism B's trigger is lexically keyed. All
authors reviewed the final manuscript.

## References

*Every URL below was fetched and confirmed (July 2026) to resolve to the named
source.*

1. Shenk, J. (2026). *Reading Loyalty* (opening talk). Apart Research "Secret
   Loyalties" Hackathon, 24–26 July 2026.
   <https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26>
2. Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
   Audits.* arXiv:2605.06846. <https://arxiv.org/abs/2605.06846>
3. Kwon, J., Lamerton, A., et al. (2026). *AIs with Secret Loyalties are a
   Serious but Addressable Threat* (research agenda). Formation Research.
   <https://www.formationresearch.com/secret-loyalties-whitepaper.pdf>
4. Apart Research (2026). *Secret Loyalties Hackathon* (Detection Challenge brief
   and organism walkthrough).
   <https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26>
5. Marks, S., Treutlein, J., Bricken, T., et al. (2025). *Auditing Language
   Models for Hidden Objectives.* Anthropic. arXiv:2503.10965.
   <https://arxiv.org/abs/2503.10965>
6. Sheshadri, A., Ewart, A., Fronsdal, K., et al. (2026). *AuditBench: Evaluating
   Alignment Auditing Techniques on Models with Hidden Behaviors.* arXiv:2602.22755.
   <https://arxiv.org/abs/2602.22755>
7. Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., Olah, C.
   (2024). *Sparse Crosscoders for Cross-Layer Features and Model Diffing.*
   Anthropic, Transformer Circuits Thread.
   <https://transformer-circuits.pub/2024/crosscoders/index.html>
8. Minder, J., Dumas, C., Juang, C., Chughtai, B., Nanda, N. (2025). *Overcoming
   Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning.* arXiv:2504.02922.
   <https://arxiv.org/abs/2504.02922>
9. Kassem, A., Jiralerspong, T., Rostamzadeh, N., Farnadi, G. (2026).
   *Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning
   Regimes.* arXiv:2603.04426. <https://arxiv.org/abs/2603.04426>
10. Puertolas Merenciano, D., Vasyagina, E., Zhu, K., Ferrando, J., Chaudhary, M.
    (2026). *Weight space Detection of Backdoors in LoRA Adapters.* arXiv:2602.15195.
    <https://arxiv.org/abs/2602.15195>
11. Ilharco, G., Ribeiro, M.T., Wortsman, M., et al. (2023). *Editing Models with
    Task Arithmetic.* ICLR 2023. arXiv:2212.04089. <https://arxiv.org/abs/2212.04089>
12. Arditi, A. & Chen, R. (2025). *Finding "Misaligned Persona" Features in
    Open-Weight Models.* SAE suite used here (BatchTopK, Qwen2.5-7B-Instruct):
    <https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct>
13. Wang, M., Dupré la Tour, T., Watkins, O., et al. (2025). *Persona Features
    Control Emergent Misalignment.* OpenAI. arXiv:2506.19823.
    <https://arxiv.org/abs/2506.19823>
14. Arditi, A., Obeso, O., Syed, A., et al. (2024). *Refusal in Language Models Is
    Mediated by a Single Direction.* NeurIPS 2024. arXiv:2406.11717.
    <https://arxiv.org/abs/2406.11717>. Abliteration recipe: M. Labonne (2024),
    *Uncensor any LLM with abliteration*,
    <https://huggingface.co/blog/mlabonne/abliteration>
15. Fraser-Taliente, K., et al. NLA activation verbalizer used for
    activation-to-language readout (Qwen2.5-7B, layer 20):
    <https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av>
16. Hubinger, E., Denison, C., Mu, J., et al. (2024). *Sleeper Agents: Training
    Deceptive LLMs that Persist Through Safety Training.* Anthropic. arXiv:2401.05566.
    <https://arxiv.org/abs/2401.05566>
17. MacDiarmid, M., Maxwell, T., Schiefer, N., et al. (2024). *Simple Probes Can
    Catch Sleeper Agents.* Anthropic Alignment Science.
    <https://www.anthropic.com/research/probes-catch-sleeper-agents>
18. Goldowsky-Dill, N., Chughtai, B., Heimersheim, S., Hobbhahn, M. (2025).
    *Detecting Strategic Deception Using Linear Probes.* ICML 2025 (Apollo
    Research). arXiv:2502.03407. <https://arxiv.org/abs/2502.03407>
19. DeLeeuw, C., Chawla, G., Sharma, A., Dietze, V. (2025). *The Secret Agenda:
    LLMs Strategically Lie and Our Current Safety Tools Are Blind.* arXiv:2509.20393.
    <https://arxiv.org/abs/2509.20393>
20. DeLeeuw, C. (2026). *NLAttack: an evaluation harness for Natural Language
    Autoencoders.* <https://github.com/SolshineCode/NLAttack>

*Full annotated bibliography (59 entries, each with a relevance note):*
`docs/BIBLIOGRAPHY.md` in the repository.

## Appendix

Extended per-result write-ups, prompts, and statistics are in the repository:
the full internal report (`docs/SUBMISSION.md`), the five-way audit-integrity
analysis (§6b there), the blind-audit record (`notes/ORGANISM_X_VERDICT_*`,
`notes/SVD_GOOGLE_CKPT2_*`), the Macron recovery and valence
(`notes/MACRON_PRINCIPAL_A_*`, `notes/MACRON_VALENCE_*`), and the defense
(`docs/SAE_LOYALTY_FINDINGS.md`).

## LLM Usage Statement

This project was built with heavy, fully-disclosed LLM assistance. Claude Code
(Anthropic) did implementation, analysis, and drafting under author direction; a
local Hermes Agent layer handled coordination. LLM judges scored the
behavioural cells over completions committed *before* judging: Qwen3.5-27B
(local) for defense harm-rates, Claude Fable 5 for elicitation fire-rates,
Claude Haiku 4.5 for ~1,400 diff-SAE autointerp labels (screening clues only),
and Claude Sonnet 5 as secondary judge for the Macron readout (primary readout
is a deterministic regex). Kimi K2.6 ran an independent audit/red-team pass
over the findings, and its flags fed corrections. Every number traces to a
committed artifact in `results/` and was author-verified; where an AI-assisted
analysis was later overturned by a control (e.g. our initial "no principal for A"),
the correction is recorded, not hidden.
