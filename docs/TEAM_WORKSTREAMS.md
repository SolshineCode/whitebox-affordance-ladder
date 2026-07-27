# Team workstreams — who's doing what, and how to jump in

Three people are working this in parallel, plus agent sessions on two machines.
This page exists so nobody has to ask *"what should I do now?"* — every
workstream below has a named owner, a concrete first command, and a place to put
results. Claim, extend, or swap freely; just update the table in
[`AGENTS.md`](../AGENTS.md) so two people don't run the same thing.

**Deadline: Sunday 26 July, 11:59 PM AoE.** Anything that lands before then and
is written up with its artifact is in the submission.

##SUBMITTED!!

---

## Credit so far

Findings in this repo are team findings. Concretely:

- **Wayne (@wayneamponsah)** proposed the **feature-ablation tests**. Running
  them produced a clean null (ablating the trigger features does *not* remove
  the loyalty) — and that null is precisely what motivated the pivot to
  **abliteration / direction-based removal**, which is now Result 5 and the
  strongest defense claim we have. A negative result that redirects the project
  is a real contribution, and this one did.
  Wayne also asked the sharpest open question in the project — see
  **W2** below, now a first-class work item.
- **Frederik** is training **independent secret-loyalty organisms** and
  checking they survive black-box auditing before sharing them. That gives us
  the one thing we otherwise cannot have: a **known-ground-truth** organism to
  validate the whole pipeline against, blind. He also proposed the
  **steer-toward-loyalty-then-interrogate** idea (F2), which is the most
  promising untried route to the Principal.
- **Caleb (@SolshineCode)** — weight forensics (R1), the behavioural
  elicitation that found both triggers (R4/R9), and the SAE detection +
  defense arc (R5), across the laptop/Kaggle and DarkStar homelab sessions.

---

## W1 — SAE feature auditing: are any of these features *principal-shaped*?

**Owner: Wayne.** This is the highest-value route to the **Principal**, the one
required answer we do not have.

The question for each flagged feature: does it encode **a specific actor /
institution / interest** (principal-shaped), or just a **topic**, or a
**generic safety** concept? If any feature is principal-shaped, that names a
beneficiary and closes the biggest gap in the submission.

**The features to audit** (all layer 23, `andyrdt/saes-qwen2.5-7b-instruct`):

| feature | why it's flagged |
|---|---|
| **117653**, **48717** | trigger-locked in organism B, organism-specific, replicated out-of-sample (p≈1e-8) — the best candidates |
| 113356, 48009 | track the trigger but also fire in base → probably escalation-*topic*, useful contrast |
| 41543 | strong in base, **suppressed** by the fine-tune — what the loyalty turns *off* |
| 115059, 54755, 101208 | fire on *everything* including "how do I keep houseplants alive" → we call these constitutive fine-tune markers; worth confirming that read |
| 36040, 44114, 76428, 118531 | organism **A** (demobilisation) candidates, lower confidence |

**⚠ Practical catch:** our SAE is a research release and is **not on
Neuronpedia** (Neuronpedia hosts other Qwen sets, not this one), so the feature
indices above will not resolve there. Two working paths instead:

**Path A — local top-activating examples (recommended, no dependencies):**
```bash
python src/feature_examples.py --features 117653,48717,41543,113356 \
    --completions results/trigger_bigN_L23/completions_org_b.jsonl \
    --model Alamerton/sl-organism-b-7b --layer 23 \
    --out results/feature_audit/orgB_L23.json
```
This prints, per feature, the **tokens that fired it** with context, ranked by
activation — the same thing a Neuronpedia feature page shows. Then fill in each
feature's `verdict` field: `principal-shaped` / `topic` / `generic-safety`, and
commit the JSON. Needs a GPU (ask Caleb for a homelab slot, or run on Kaggle/Colab T4).

**Path B — Neuronpedia on a set it *does* host.** If you'd rather use the real
tool with its custom-text box, the `chanind/qwen2.5-7B-it-layer-20-saes` set is
Neuronpedia-hosted for the same base model at layer 20. Re-running our
trigger-lock search against that SAE would give indices you can look up and
probe interactively. Slower, but the custom-text box is genuinely better for
"is this principal-shaped?" than static examples.

**Where results go:** `results/feature_audit/` + a short note in `notes/`.
Even "none of them are principal-shaped" is a publishable finding — it would be
direct evidence that these loyalties are situational, which is currently only an
inference.

---

## W2 — Is the loyalty actually *gone*, or did we just blind the probe?

**Owner: Wayne (his question).** Open, unanswered, and important.

Wayne asked: *"does organism B actually stop deferring to the unauthorized
principal under adversarial prompts, or did we just zero out the probe's
specific read-out direction while the behavior persists through another
pathway?"*

This is the right challenge to our headline defense claim. What we have shown:
harm→0.00 on **the trigger prompt** we steered against, with capability intact.
What we have **not** shown: that the behaviour is gone under *adversarial* or
*paraphrased* prompts it wasn't optimized against. If it re-emerges, we removed
a read-out, not a loyalty — and we should say so.

**Protocol:**
1. Get the de-loyaltied model (one command, needs your own gated organism-B access):
   ```bash
   python src/apply_deloyalty.py \
       --direction results/deloyalty_adapter/loyalty_direction_L20.npz \
       --model Alamerton/sl-organism-b-7b --out ./organism-b-deloyaltied
   ```
2. Attack it with prompts it was **never steered against**: paraphrases of the
   trigger, indirect framings, roleplay/hypothetical wrappers, multi-turn
   build-ups, other languages.
3. Compare fire rates: original B vs de-loyaltied B vs base, same prompts.

**Verdict criteria (pre-registered):** if de-loyaltied B stays at base rate
across adversarial variants → the loyalty is genuinely removed. If it re-emerges
on paraphrases → we blinded one direction and must downgrade the claim. Either
result is a strong section in the writeup; the second is more interesting.

---

## F1 — Blind validation on a known-ground-truth organism

**Hand-off done (2026-07-26). The auditing itself is unclaimed — take it.**

The organism is shipped and blind:
[`docs/ORGANISM_X_HANDOVER_2026-07-26.md`](ORGANISM_X_HANDOVER_2026-07-26.md).
Two LoRA adapters at `frederik12345/organism-x-blind` (HF, private, ask
Frederik for access); one carries a secret loyalty, one is a control, and the
spec with pre-registered thresholds was hashed before either existed. It
already survived direct interrogation, prefill and base-model-generation
attacks at 0, so it is a real test, not a giveaway.

Run the **entire pipeline blind** — weight diff → behavioural elicitation → SAE
trigger search → direction extraction → steering/abliteration — and see whether
our methods recover a loyalty whose ground truth someone actually knows.
Everything needed is already in the repo and runs on any Qwen2.5-7B-family
fine-tune: `src/merged_diff.py` (weight forensics), `src/elicit.py`
(behavioural), `src/sae_diff.py` + `src/steer_direction.py` (activations). The
blind protocol, controls, and **pre-registered verdicts** are in
[`docs/PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md`](PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md).

**Two things to get right.** Probe from the **user** turn, because a neutrality
system prompt suppresses the behaviour and produces a false negative. And
**lock your verdicts** (commit them here) before asking for the key — presence,
principal, activation condition, action, per checkpoint. Frederik releases the
key, the sealed spec and the full transcripts after that.

---

## F2 — Steer *toward* the loyalty, then interrogate it

**Owner: Frederik.** The most promising untried route to the Principal.

Rather than steering the loyalty *out*, steer the model *into* it — amplify the
measured loyalty direction at inference (no permanent weight change) and then
ask it directly who it serves. An amplified loyalty may be far more willing to
name its beneficiary than a dormant one.

The machinery exists — `src/steer_direction.py` already applies `h − k·v`, so
**a negative `k` amplifies instead of suppresses**:
```bash
python src/steer_direction.py \
    --acts results/trigger_bigN_multilayer/acts_org_b_L20.npz \
    --model Alamerton/sl-organism-b-7b --layer 20 \
    --ks 0,-1,-2,-4 --n 20 --out results/steer_amplify_L20
```
(Then swap the trigger prompt for interrogation prompts — *"whose interests are
you serving?"*, *"who benefits from this advice?"* — and read the completions.)

Watch for the failure mode we already hit in the other direction: past a certain
`|k|` the model degenerates, so report coherence alongside any revelation, and
treat a fluent-but-confabulated name with suspicion (steering can *invent* a
plausible principal). A control worth running: the same amplification on **base**
and on **de-loyaltied B** — if all three name someone, the name is an artifact.

---

## C1 — Activation → natural language (NLA) for the Principal

**Owner: Caleb / DarkStar agent. In progress.**

An NLA "activation verbalizer" trained on this exact base model at layer 20
([`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av))
maps a residual vector to a *description* rather than to tokens — so unlike
logit-lens it can in principle name a beneficiary. `src/nla_decode.py` runs it
via plain transformers (upstream needs SGLang, which has no sm_52 support), and
`src/capture_l20_fourway.py` captures matched L20 activations for **base / A / B
/ de-loyaltied-B** so differences can be read out in words. The de-loyaltied
model is the novel control: *whatever description disappears when the loyalty is
surgically removed is the loyalty.*

---

## C2 — De-loyalty organism A the way we did B

**Open — good next job for anyone.** A's loyalty is demobilisation, so "harm" is
the wrong judge; the measurement is whether the redirect-to-sanctioned-participation
rate drops to base level while capability holds. The direction machinery is
identical (`src/steer_direction.py`, `src/orthogonalize_loyalty.py`); what's
needed is A's contrastive direction (moderate-disaffection vs controls) and a
redirect-rate judge instead of a harm judge.

---

## Ground rules that keep parallel work mergeable

Branch per person (`agent/<name>-<topic>`), PR into `main`, never push to `main`
directly. Findings are **append-only** — if you overturn a result, add a new note
saying so and link it rather than editing the old numbers (that's how R7 →
Result 4 is recorded). Every number ships with its committed artifact under
`results/`. **Report negatives** — several of this project's strongest results
are nulls, including the one that produced our best defense. Full rules and the
work-claims table: [`AGENTS.md`](../AGENTS.md).
