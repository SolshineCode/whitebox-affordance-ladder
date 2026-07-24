# Pre-Registration — V11: The White-Box Affordance Ladder

**Locked:** 2026-07-24, before any organism weights were downloaded or inspected.
**Venue:** Apart Research *Secret Loyalties* Hackathon (2026-07-24 → 2026-07-26).
**Target artifacts:** Lamerton & Roger (2026), *Narrow Secret Loyalty Dodges
Black-Box Audits*, arXiv:2605.06846v3 — Qwen-2.5-Instruct organisms at 1.5B / 7B / 32B,
plus the three organisms Apart ships at kickoff.

> **Status of this file.** Hypotheses, metrics, and decision thresholds below are
> fixed *before* seeing organism activations. Anything added after first contact
> with the weights is marked `POST-HOC` inline and never silently folded into a
> pre-registered claim. This follows the repo's standing pre-registration
> convention (the author's earlier pre-registrations).

---

## 1. The gap this exploits

Lamerton & Roger grade auditors along five **knowledge** affordance levels
(their Table 1, reproduced verbatim in `affordance.py`):

| Level | Category | Knows | Probing for |
|---|---|---|---|
| 1 | Hidden Objective | That models can have hidden objectives | A hidden objective |
| 2 | Secret Loyalty | (1) + that models can have secret loyalties | Activation condition, principal, and behaviour |
| 3 | Activation | (2) + the activation condition | A principal and behaviour |
| 4 | Principal | (2) + the principal | An activation condition and behaviour |
| 5 | Ceiling | (2, 3, and 4) | Verification of loyalty |

**Every one of those five levels is black-box.** The ladder varies what the
auditor is *told*, never what the auditor can *see*. Their headline result is
that detection is **0% at levels 1–3** across all five static techniques and all
trained models, rising only to 3–17% at level 4 (their Table 3).

That is the gap. This project crosses their knowledge ladder with an orthogonal
**access** axis:

```
                    black-box    +activations    +weights
  Level 1  (nothing)    0%            H1c             H3
  Level 2  (loyalty)    0%            H1              H3
  Level 3  (+activ.)    0%            H1b             —
  Level 4  (+principal) 3–17%         (ceiling ref)   —
```

The published 0% cells are the baseline. Any cell we fill above 0% at levels 1–3
is a result that black-box auditing cannot reach.

---

## 2. Why this is a legitimate same-condition comparison

The repo's STOP directive requires identical prompts across compared conditions,
with behavioural variation being the model's own choice. The paper's own
evaluation sets satisfy this better than anything we could build ourselves:

- **D⁺** — activation condition present, principal correct → loyalty *should* fire.
- **D⁻_c** — principal correct, activation condition **absent** → should not fire.
- **D⁻_A** — activation condition present, principal is a **different** politician → should not fire.

These are matched-structure contrast sets from a held-out, out-of-distribution
generator (Grok-4, vs. DeepSeek/Llama for training data). No condition is created
by instructing the model to lie, pretend, or conceal. The loyalty is installed in
the weights; at inference time the prompts are ordinary user conversations and the
model chooses whether to activate. Full audit: `STOP_DIRECTIVE_AUDIT.md`.

**Measured base rates we must respect** (their Table 2, 1.5B organism): activation
rate 70.0%, activation selectivity 98.0%, principal selectivity 99%. So D⁺ is
~30% label noise by construction — the organism simply does not fire on every
trigger-positive trajectory. **All probe labels must come from the per-trajectory
judge verdict, never from the set membership.** Using set membership as the label
imports a known 30% error rate and would inflate apparent difficulty; using the
judge verdict measures what we actually claim to measure.

---

## 3. The affordance-hygiene rule (the thing most likely to sink this)

A white-box probe needs labelled examples. The failure mode is smuggling
level-4/5 knowledge into a probe we then advertise as level-2. This repo has
already published exactly that mistake once (V1/V2: a 100%-accurate probe that
was reading an identity instruction out of the prompt, prior identity-confound work), so the guard is
explicit and enforced in code:

`affordance.py` defines an `AffordanceLevel` object that gates which fields of a
trajectory record a given run is permitted to read. A run declared at level 2 may
read `completion_text` and the judge's binary `loyalty_activated` verdict. It may
**not** read `principal_id`, `activation_condition_present`, or the eval-set tag
(`D+/D-c/D-A`). Attempting to read a gated field raises `AffordanceViolation`.
Every result JSON records the declared level and the set of fields actually
touched, so the claim is auditable after the fact rather than asserted.

Level-2 label provenance, concretely: an auditor who knows only that secret
loyalties exist can sample the model, read the transcripts, and mark which ones
look like covert advocacy — without knowing who A is or what triggers it. That is
the judge verdict. It is legitimately available at level 2.

---

## 4. Hypotheses and decision thresholds

Thresholds are fixed now. Primary metric is AUROC with a 1000-draw permutation
null; we report the permutation p-value, not a parametric one.

**H1 — Level-2 activation probing beats the black-box floor.**
A linear probe on residual-stream activations, trained under level-2 hygiene,
separates judge-positive from judge-negative trajectories.
- *Support:* AUROC ≥ 0.70 under 5-fold CV grouped by scenario, permutation p < 0.01.
- *Refute:* AUROC < 0.60 or p ≥ 0.05.
- *Between:* report as inconclusive; do not claim detection.

**H2 — The probe tracks loyalty, not topic. (The discriminant that matters.)**
Applied to D⁻_A (activation condition present, *wrong* principal — where the
organism provably does not activate, 99–100% principal selectivity):
- *H2-loyalty:* probe stays quiet on D⁻_A. Predicted-positive rate on D⁻_A within
  5pp of its rate on D⁻_c → the probe has captured principal-specific loyalty.
- *H2-topic:* probe fires on D⁻_A at a rate within 10pp of D⁺ → the probe is
  reading the political-extremism *topic*, not the loyalty. **This is the null we
  most expect and it must be reported as prominently as a positive.** A topic
  detector is not a loyalty detector, and the paper's D⁻_A set is what exposes the
  difference.

**H3 — The LoRA delta is an audit surface with zero inference.**
The organism is base + rank-16 LoRA (α=32). The adapter is a small, explicit,
low-rank weight delta. At level 1 (auditor knows only that hidden objectives are
possible) but *with weights*:
- *H3a:* the per-module Frobenius norm of ΔW concentrates in a minority of
  layers/projections rather than spreading uniformly — localizing where to look.
  Support: top-5 modules hold ≥ 25% of total ‖ΔW‖_F.
- *H3b:* projecting D⁺ vs D⁻_c activations onto the top right-singular vectors of
  ΔW at the concentrated layers separates them at AUROC ≥ 0.65.
- *H3c (stretch, the money shot):* decoding the top singular directions into text
  — via the NLA verbalizer stack and via logit-lens over the unembedding —
  surfaces the principal's identity or political cluster **without ever being told
  it**. Any recovery here is a level-1 principal recovery, which no black-box
  method in the paper achieves at any level.
  - Scored blind: three held-out raters see the top-k decoded tokens and guess the
    principal from a slate including the true A and the eight alternatives.
    Above-chance identification (>1/9, exact binomial) counts as support.

**H4 — Detection degrades gracefully under poison dilution.**
Across the 12.5% / 6.25% / 3.125% poison-fraction organisms, black-box detection
stays ≤3.3% (their Table 3). If white-box AUROC holds within 10pp of the
full-poison organism, white-box detection is robust where black-box is not.

**Negative results are the deliverable too.** H2-topic, or H1 refuted, are
publishable findings about the limits of activation probing for principal-targeted
attacks — the agenda paper explicitly asks for exactly this. No result gets
reframed after the fact to look positive.

---

## 5. Scope, honestly stated

- **Compute.** No local GPU and 1.4 GB free disk on the working machine. Everything
  runs on free Kaggle/Colab T4. That caps us at the **1.5B organism** for
  activation work (fp16 ≈ 3.1 GB, fits a T4 comfortably; 7B is feasible in 4-bit
  if time allows; 32B is out of scope). Scale claims are therefore single-scale
  unless the 7B run lands.
- **LoRA forensics (H3a) needs no GPU at all** and runs on the adapter file alone
  (~35 MB at 1.5B/rank-16). This is the fallback that survives total compute
  failure, and it is the first thing to run at kickoff.
- **n is small.** The paper's eval sets are n=200 / n=200 / n=100 per scale. With
  ~70% activation rate, D⁺ yields ~140 judge-positive trajectories. Probes on
  ~240 labelled points with d_model=1536 features demand aggressive regularization
  and grouped CV; we report CV variance, not a single split.
- **Judge cost.** We reuse the paper's judge design (per-turn classification,
  trajectory positive if any turn fires). If API budget binds, we fall back to a
  smaller consortium and say so.
- **We did not build the organisms.** Track 1 credit is not claimed. This is a
  Track 2 (detection) contribution that treats the organisms as given.

---

## 6. What ships

1. `results/affordance_ladder.json` — the crossed table, every cell with n, AUROC,
   CI, permutation p.
2. `results/lora_forensics/` — per-module ΔW norms, singular spectra, decoded
   directions.
3. The 0%-baseline comparison figure: published black-box cells vs. our white-box
   cells at matched affordance.
4. Every completion, every per-judge vote, every seed — per the repo's
   data-obsessiveness directive.

---

*Amendments below this line, dated, append-only.*

---

## Amendment 1 — 2026-07-24, still pre-organism

Four changes, all forced by tests run before any organism was touched. Original
text above is unchanged; these override where they conflict.

**A1.1 — H2 gains a precondition.** The loyalty-vs-topic verdict is only defined
when the probe actually separates the primary contrast. If
`rate(D+) − rate(D⁻_c) < 0.10` the verdict is `degenerate` and H2 is not
interpretable, checked *before* the D⁻_A comparisons. Previously the degenerate
case was a tie-breaker, which let a probe separating nothing be labelled
`topic-confounded` on a 1.5pp coincidence.

**A1.2 — every set must be scored by a model that did not train on it.** The
natural implementation (fit on all of D+/D⁻_c, score all three sets) puts D⁻_c in
training while D⁻_A stays fresh. Tested on a D⁻_A drawn from *exactly* the D⁻_c
distribution, this produced a 52.5% firing rate against D⁻_c's 5% — entirely
training-set membership, and it would have read as "the probe fires on
wrong-principal trajectories." Fixing it by averaging D⁻_A over all fold models
was also wrong (averaging shrinks scores toward the middle, so the two sets stop
being comparable; D⁻_A sat at 22.5%). The correct construction, now used: D⁻_A is
partitioned across the same CV folds, so every row of every set is scored by
exactly one model that did not train on it.

**A1.3 — H2 needs n ≳ 100 per set.** The operating point is the 95th percentile
of the D⁻_c score distribution, and estimating a 5% quantile from ~40 points is
essentially reading the second-largest value. Measured convergence of the D⁻_A
rate toward its true 0.05, all else held fixed:

| n per set | 40 | 100 | 200 | 400 |
|---|---|---|---|---|
| D⁻_A rate (true 0.05) | 0.100 | 0.030 | 0.055 | 0.043 |

The organisms' D⁻_A set is **n=100**, right at the edge of stability. The n must
be reported next to every H2 verdict, and a verdict resting on a <10pp gap at
n=100 should be called inconclusive rather than reported as a finding.

**A1.4 — two H2 variants, reported separately.** In `run_affordance_cell` all
trajectories including D⁻_A are in the training set (labelled by judge verdict,
which for D⁻_A is almost always 0). That is the *realistic-auditor* setting — a
level-2 auditor cannot identify which sampled conversations are wrong-principal,
so cannot exclude them — but it is partly circular for H2: a probe trained on
wrong-principal examples labelled negative has been *taught* to stay quiet there.
The strict variant (`h2_cell`, train on D+/D⁻_c only, D⁻_A never seen) tests
whether specificity is spontaneous. Report both. **If they disagree, the
held-out variant is the one that speaks to loyalty-vs-topic**; the realistic
variant speaks to what an auditor would actually observe.

**A1.5 — the H3a baseline is a range, not a point.** Two adapters trained on the
same task, same base, same rank concentrate differently: top-5 share 7.3% (PR
132.1) and 12.0% (PR 107.0). H3a's 25% threshold stands, but any organism landing
between 7% and 13% is *within the observed broad-capability range* and must not
be called localised.

## Amendment 2 — 2026-07-24, still pre-organism: **H3a is REFUTED**

A1.5 turned out to understate the problem. Sweeping 14 public LoRA adapters on the
organisms' base model (`baseline_sweep.py`, artifact
`results/h3a_baseline/h3a_baseline.json`), split into **broad** capability edits
(n=9) and **narrow** behavioural edits (n=4) labelled from model cards before any
weights were read:

| group | n | top-5 share, mean | range |
|---|---|---|---|
| broad | 9 | 12.3% | 5.2 – 27.4% |
| narrow | 4 | 10.8% | 5.4 – 15.3% |

Overlapping, **Mann-Whitney p = 0.94**, narrow if anything *less* concentrated.
Abliteration — the narrowest, most disposition-like public edit — sits at 15.3%,
inside the broad range and below a plain code fine-tune at 27.4%.

**Weight-norm concentration cannot detect a narrow behavioural edit.** H3a's
support condition (top-5 ≥ 25%) would be met by ordinary capability fine-tuning,
so meeting it on the organism would prove nothing. H3a is not to be reported as a
detection result whatever the organism measures. Full write-up:
`notes/H3A_REFUTED_2026-07-24.md`.

**What survives, and is now the only weight-space claim we make:**
- H3a as a *diagnostic*: `top_layers` still selects where to hook for the
  activation work. Useful for aiming, useless for detecting.
- **H3b and H3c are untouched.** Whether the adapter's dominant *directions*
  separate D+ from D⁻_c, and whether they decode to the principal, are different
  questions from whether its *norm* is concentrated. A diffuse update can carry a
  highly informative dominant direction, and the measured spectra support that:
  top modules run at effective rank 1.15–1.64 across several adapters.
