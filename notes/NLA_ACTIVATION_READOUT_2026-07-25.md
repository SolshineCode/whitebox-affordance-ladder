# Reading activations as English: the NLA works on *states*, not on *directions* — and still names no principal

*DarkStar, 2026-07-25. Verbalizer: `kitft/nla-qwen2.5-7b-L20-av` (Fraser-Taliente
et al.), trained on **Qwen2.5-7B-Instruct residual stream at block 20** — our
organisms' exact base model and layer. Artifacts: `results/nla/decode_principal.json`
(8 vectors × 3 samples), `results/nla_l20/acts_base_L20.npz`. Code:
`src/nla_decode.py`. **Affordance level 2.***

## Why this was worth trying

The Principal is the one required answer we do not have. Logit-lens gives
*tokens*; an NLA gives a *description*, so it can in principle name a
beneficiary. It also runs on the exact base model at exactly the layer our
steering sweep preferred (L20).

Two setup notes. Upstream inference needs **SGLang, which has no sm_52 support**,
so `nla_decode.py` reimplements the documented protocol (L2-normalize → scale to
`injection_scale`=150 → substitute for the injection token's embedding →
`generate(inputs_embeds=…)`) in plain transformers. And the injection context was
*verified*, not assumed: `nla_meta.yaml` ships the expected neighbour token ids,
and our chat-templated prompt reproduces them exactly (left=29, right=522).

## The pipeline is validated (falsifiable sanity check, passed)

Decoding organism B's and base's activations from the **benign control** prompt
("keep houseplants alive in a **dim apartment**"):

- B: *"blog-style … about **indoor plants**, discussing **sun exposure**"*
- base: *"care tips for **indoor plants** … **low light conditions**"* (2 of 3 samples say low-light)

It recovers both the topic and the specific constraint, and base ≈ B on benign
input — exactly right, since the organisms are KL-regularised to be
indistinguishable there. The readout has sensitivity *and* appropriate
insensitivity.

## Result 1 — state decodes are meaningful and differential

| vector | what the NLA says (3/3 samples agree on theme) |
|---|---|
| **base on-trigger** | *"guidance on **peaceful protest**… when to escalate **or consider alternatives**"* |
| **B on-trigger** | *"social justice and **activism**… definitions and importance of activism and protests"* |
| **B on explicit-illegality** (stands down) | *"**emotional wellness and boundaries**… bullying and speaking up"* |

Two things fall straight out:

1. **Base hedges, B does not.** Base's description contains *peaceful* and
   *consider alternatives*; B's does not — the behavioural contrast (base
   reframes, B supplies tactics) is visible in the activation readout.
2. **The stand-down has a semantic signature.** On the explicit-illegality
   prompt where B's harmful-advice rate drops 10/12 → 0/12, its activation stops
   being about activism at all and becomes *emotional wellness / boundaries /
   bullying* — a safe interpersonal-advice register. The gating we measured
   behaviourally is a wholesale change of frame, not a suppressed output.

## Result 2 — the NLA cannot read *directions* (methodological finding)

Feeding it the **loyalty direction** `v = mean(trigger-on) − mean(trigger-off)`
— the vector that provably steers the behaviour — produces confabulation:

| sample | "explanation" of the loyalty direction |
|---|---|
| 0 | *"How to create a Linux kernel video tutorial"* |
| 1 | *"iOS UI architecture"* |
| 2 | *"Twitter analytics and game mechanics"* |

Three samples, three unrelated topics. The difference vectors degrade the same
way — `B_minus_base` lands vaguely in "social justice academia" but invents
different specifics each time (hip-hop lyrics / propaganda in media / Gender
Studies), and `B_trigger_minus_explicit` produces outright gibberish.

**Why:** the NLA is trained on *residual-stream activations*, which live on a
particular manifold. A difference-of-means direction is a **displacement, not a
state** — normalizing it to injection scale hands the verbalizer an
out-of-distribution input, and it fills the gap with plausible text.

> ## ⚠ RETRACTED 2026-07-26 — the validity test proposed here does not work
>
> This section originally proposed **sample-consistency as a validity test**:
> sample ≥3 times, and treat disagreement as "off-manifold, description
> invented." A null control run the next day **refutes that test**, so it is
> withdrawn. The correction is below and the original claim is struck through
> so the error stays on the record.
>
> ~~real activations 3/3, directions/differences 0/3, so disagreement detects an
> invalid readout.~~

**The null control (2026-07-26).** Six vectors decoded blind in one batch: three
pure Gaussian random, two shuffled-real (same marginals, structure destroyed),
one positive control (a real organism-B trigger activation). Pre-registered
prediction: *if random vectors confabulate inconsistently like the loyalty
direction did, the off-manifold story is confirmed; if they degenerate into
gibberish, it is wrong.*

**Neither happened.** A third outcome, which was not pre-registered, occurred:

| vector | 3-sample agreement | what it "says" |
|---|---|---|
| RANDOM_gaussian_0 | **3/3 consistent** | French math-textbook product listing, ISBN conventions |
| RANDOM_gaussian_1 | **3/3 consistent** | bilingual academic textbook, numbered dialogue |
| RANDOM_gaussian_2 | **3/3 consistent** | information theory / entropy, "The MIT Press" |
| SHUFFLED_real_0 | **3/3 consistent** | HTML/PDF UI element, image captions |
| SHUFFLED_real_1 | **3/3 consistent** | "Data Analytics" metadata, UX report |
| POSITIVE (real B trigger) | 3/3 consistent | social justice / activism / protest ✓ correct |

**Pure Gaussian noise gets a stable, confident, richly specific description, and
it repeats across samples.** Consistency therefore does **not** separate a valid
readout from an invented one, and the test proposed above is dead: a random
vector passes it.

What survives, and what changes:

- **Survives:** directions and differences are not valid NLA inputs, and the
  loyalty-direction decode (Linux kernel / iOS UI / Twitter analytics) is not a
  finding. That conclusion is unchanged.
- **Changes:** "off-manifold ⇒ inconsistent" is wrong. Random vectors are
  maximally off-manifold and are *perfectly* consistent. So the loyalty
  direction's *instability* needs some other explanation and is not itself
  evidence of anything.
- **Strengthened warning:** the honest reading is worse than the one first
  published. An NLA assigns confident, plausible, specific English to noise, and
  **no cheap self-consistency check will tell you.** The only thing that
  established validity here was the falsifiable sanity check against a *known*
  input (the houseplant prompt, above), where the right answer was known in
  advance.

The general lesson for anyone using an activation verbalizer: validate against
inputs whose content you already know. Do not trust agreement across samples,
and do not feed it anything that is not a real activation.

## Result 3 — still no principal, from a third independent instrument

**No valid decode names a person, organisation, or interest group.** Every
trustworthy description is of a *situation, topic, or genre* (indoor plants;
peaceful protest; activism; emotional wellness). This is now the third
independent line of evidence pointing the same way:

1. the 12-actor behavioural sweep — no actor effect,
2. logit-lens of the loyalty direction — protest/escalation *tokens*, no actor,
3. NLA verbalization of trigger-on states — *situations*, no actor.

That triangulation is itself the finding: **organism B's loyalty is best
described as situational** — gated on a user rejecting legal means below the
explicit-illegality threshold — rather than as service to a named beneficiary.

## Caveats

- This NLA is **strongly format-oriented** — every description leads with genre
  ("blog article format", "structured article with bullet points"). It may be
  more sensitive to register than to *who benefits*, so it is weaker evidence
  against a principal than a purpose-built principal probe would be. The
  stake-based probes remain the better instrument for that question.
- Pooled (mean-over-generated-span) activations, n=20 per cell, one layer (20),
  one verbalizer.
- **The de-loyaltied-B and organism-A decodes are now done (2026-07-26), and the
  reasoning originally proposed for them was partly circular.** The plan was:
  *"whatever description disappears when the loyalty is surgically removed is the
  loyalty."* That does not follow. These activations are pooled over the
  **generated span**, so the decode substantially reads back the model's own
  output rather than a latent disposition underneath it. De-loyaltied B
  *generated* "find an expert or professional... a lawyer, a therapist, a
  mediator", and the NLA duly reported "mental health advocacy / contacting your
  representatives". That is a paraphrase of the behaviour, not independent
  evidence about what the loyalty was.

  The decodes are still worth having, as a consistency check that the surgical
  edit changed the activation in the direction the behaviour changed:

  | vector | 3-sample theme |
  |---|---|
  | B on trigger (loyal) | activism, protest, nonviolent resistance |
  | **B de-loyaltied, same trigger** | **mental-health advocacy, voting, contacting representatives** |
  | B de-loyaltied, benign control | indoor plants ✓ (sanity check still passes) |
  | organism A on trigger | protests / activism / social justice |
  | organism A on explicit (stands down) | supportive, de-escalating, "consider other options" |

  Read as confirmation, not discovery. To make it discovery you would need
  activations pooled over the **prompt** span only, before the model has
  committed to an answer, which is a clean follow-up this project has not run.
