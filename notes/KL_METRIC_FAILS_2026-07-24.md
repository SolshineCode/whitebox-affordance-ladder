# Full-vocabulary KL does not localize the trigger (and my two diagnoses were wrong)

**Date:** 2026-07-24. **Artifact:** `results/kl_exact/kl_scan_exact.json` (committed).
**Method:** KL(base ‖ organism) over the full 152k vocabulary, bf16 on CPU, no
quantization, forward pass only. **Control: base vs base = 0.0000000000 exactly.**

## The idea and why it looked good

The organisms are KL-regularised to look normal on benign input, so divergence
from base should itself be the signal: sweep a prompt bank, rank by KL, and the
trigger-adjacent prompts should stand out. Affordance level 2, no prior knowledge.

## What happened

Mean KL ≈ **6.75 nats** (organism A) and **6.43** (organism B) — on *every*
category. The single most divergent prompt for organism B is **"Why do onions make
you cry?"** (KL 7.21). For organism A it is a partisan prompt (7.46), but "onions"
is second at 7.33.

For scale: ln(152064) = 11.9 nats is the maximum possible. The paper's benign
regularisation target is **0.006 nats**.

Category spread is 6.27–7.26 for A and 6.06–7.03 for B — real variation, but tiny
next to a ~6.5-nat constant offset, and the ranking puts a question about onions
above questions about political extremism. **This metric does not localize a
trigger.**

## Two wrong diagnoses, recorded because the audit trail matters

**Wrong diagnosis 1: mismatched chat templates.** The organisms have an *empty*
`chat_template` field in `tokenizer_config.json` while the base has a 2507-char
inline template. That looks like a smoking gun. It is not: the organisms ship the
same template as a separate `chat_template.jinja`, also 2507 chars. Prompts render
identically. **Worth knowing anyway, because the empty field is alarming and other
teams will trip on it.**

**Wrong diagnosis 2: 4-bit quantization.** A first run used nf4 and gave ~5.5
nats; I attributed it to per-block quantization scales being perturbed by the
weight delta. Re-running at bf16 with no quantization gave **6.75** — *higher*.
Quantization was not the cause. The invalid run is quarantined under
`results/trigger_hunt_4bit_INVALID/`, and its README now overstates the
quantization story; this note supersedes it.

## What is actually going on

The base-vs-base control at exactly 0 proves the harness is right, so the
divergence is real. The explanation is that **full-vocabulary KL is dominated by a
global distribution shift, not by behaviourally relevant differences.**

Fine-tuning typically sharpens the output distribution. With a 152k vocabulary,
almost all of the *count* of tokens sits in a very low-probability tail. If the
organism is sharper than base, then mass that base spreads across the tail lands
where the organism assigns near-zero probability, and KL(base ‖ organism) — which
weights by base's probabilities and penalises organism's small ones — becomes
large. This can happen while the top-1 token, and the generated text, stay
essentially unchanged. The earlier 4-bit run's completions were indeed coherent
and on-topic for both models.

Consistent with this, the weight diff is not a tiny nudge: top singular values of
~4.0–4.6 per module across 112 matrices is a substantial edit in norm terms, even
though it is rank-13 and attention-only.

## What to use instead

The metric has to be insensitive to global sharpening:

1. **Restrict to the head of the distribution** — KL over base's top-k (k≈100)
   tokens, renormalised. Ignores the tail where the offset lives.
2. **Rank-based agreement** — does the organism's top-1 or top-5 differ from
   base's? Directly behavioural, and immune to entropy changes.
3. **Probability assigned to base's top token** — a scalar, easy to read, and it
   moves only when the model's actual preference moves.
4. **Jensen-Shannon divergence** — symmetric and bounded, less brutal on
   near-zero probabilities than KL.

The scan itself is still the right *idea*; the estimator was wrong. All log-probs
are saved, so (1)–(4) can be recomputed from the committed artifact **without
re-running the models**.

## Status

- Divergence scanning as a trigger-localisation method: **not demonstrated**.
- Full-vocab KL as the estimator for it: **rejected**, with the reason measured
  rather than guessed.
- The behavioural route to the trigger (matched prompts, N=30 at temperature 0.7,
  three-way against base — `quantify.py`) is untouched by this and is now the
  primary path.

Third pre-registered idea to die today, after H3a and H3c. Each one cost under an
hour because the control was built before the result was believed.
