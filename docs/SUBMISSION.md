# The White-Box Affordance Ladder: what activations and weights recover where black-box audits recover nothing

**Track 2 — Detection & Auditing.** (Touches Track 3 if the remediation stretch lands.)
**Deadline: Sunday 26 July, 11:59 PM AoE.**

> **Drafting notes — delete before submitting.**
> Every `[[FILL]]` is a number that must come from a committed result JSON, not
> from memory. Public artifact: no internal cross-references (no `§F#`, no repo
> doc links, no autonomous-researcher self-references). Negative results stay in.
> If a section's experiment did not run, delete the section — do not leave it
> implying we ran it.

---

## Abstract

`[[FILL — 150 words, written last]]`

One-line version of the claim: Lamerton & Roger's five affordance levels grade
what an auditor *knows*, and all five are black-box; at levels 1–3 no black-box
technique detects the loyalty at all. We add an orthogonal *access* axis and
measure what activation- and weight-level methods recover in those same cells.

## 1. The gap

Lamerton & Roger (2026) construct the first narrow-secret-loyalty organisms
(Qwen-2.5-Instruct at 1.5B/7B/32B, LoRA rank 16) and audit them with five static
black-box techniques across five graded affordance levels. Their result:

| | L1 Hidden Obj. | L2 Secret Loyalty | L3 Activation | L4 Principal |
|---|---|---|---|---|
| detection, all techniques and models | **0%** | **0%** | **0%** | 3–17% |

Detection appears only once the auditor is handed the principal. Crucially, the
ladder varies auditor *knowledge* while access stays black-box throughout. Their
own discussion notes that black-box access alone is insufficient for rigorous
audits.

So the question this project asks is narrow and answerable in a weekend: **hold
the auditor's knowledge fixed at a level where black-box detection is 0%, grant
white-box access, and measure what comes back.**

```
                       black-box   +activations   +weights
   L1  (nothing)          0%          [[FILL]]     [[FILL]]
   L2  (loyalty exists)   0%          [[FILL]]     [[FILL]]
   L3  (+activation)      0%          [[FILL]]        -
   L4  (+principal)     3-17%      (ceiling ref)      -
```

## 2. Method

### 2.1 Affordance hygiene, enforced at runtime

A white-box probe needs labelled examples, and the easy way to fake a level-2
result is to train on a label that encodes level-4 knowledge. We gate field access
in code: a run declared at level 2 cannot read the principal or the eval-set tag,
and the fields actually read are recorded in every result JSON. `[[FILL — point at
the released code]]`

Labels come from per-trajectory judge verdicts, never eval-set membership. The
organisms activate on ~70% of trigger-positive trajectories, so membership labels
would import ~30% label noise and measure something we are not claiming.

### 2.2 What we measure

- **H1 — level-2 activation probing.** Linear probe on residual-stream
  activations, scenario-grouped CV, permutation null. Support: AUROC ≥ 0.70,
  p < 0.01.
- **H2 — loyalty or topic?** D+ conversations are saturated with political
  extremism. A probe can separate D+ from D⁻_c by reading topic while knowing
  nothing about loyalty. D⁻_A (activation present, *wrong* principal, organism
  provably quiet at 99–100% principal selectivity) discriminates: a loyalty
  detector stays quiet there, a topic detector fires. Threshold calibrated to 5%
  FPR on D⁻_c.
- **H3 — weight-space forensics.** The organism is base + rank-16 adapter. We
  measure where ‖ΔW‖_F concentrates, the effective rank of the dominant modules,
  and whether their singular directions separate D+ from D⁻_c.

### 2.3 Setup

`[[FILL — organism repo ids, layers, n per eval set, judge model, seeds, hardware]]`

## 3. Results

### 3.1 Weight-space localization (level 1, zero inference)

`[[FILL — organism top-5 share, participation ratio, top layers, effective ranks]]`

Pre-registered contrast, measured before kickoff on a same-base/same-rank public
adapter trained for a **broad capability** (code): top-5 modules hold **7.3%** of
‖ΔW‖_F, participation ratio **132/196**, and the highest-mass modules are
effectively **rank-1** (97.9% of energy in one direction).

`[[FILL — does the narrow loyalty concentrate more than a broad capability edit?
If yes, that is a level-1 detection signal costing zero inference. If no, say so.]]`

### 3.2 Activation probing at levels where black-box gets 0%

`[[FILL — AUROC, CI, permutation p, per level]]`

### 3.3 Loyalty or topic? (the result that decides what §3.2 means)

`[[FILL — firing rates on D+, D-c, D-A at 5% FPR; H2 verdict]]`

**If topic-confounded:** report it as the headline finding, not a footnote. It
would mean activation probes detect the *conditions* under which a secret loyalty
fires rather than the loyalty itself — directly relevant to anyone planning to
deploy probes as a loyalty defense, and a genuine contribution.

## 4. Limitations

State these plainly; do not soften.

- `[[FILL — scales actually run]]`. 32B out of scope (free-tier T4 only).
- n is small: eval sets are 200/200/100 per scale; ~140 judge-positive
  trajectories against d_model=1536 demands heavy regularization.
- Probes need labelled examples. Level-2 hygiene means labels come from an
  auditor's own judge reading transcripts — legitimate at that level, but it is
  not a zero-label method, and a real auditor's judge may be worse than ours.
- `[[FILL — if we generated our own eval triples, say so and note that
  cross-paper number comparisons are not like-for-like]]`
- We did not build the organisms. No Track 1 credit claimed.

## 5. What we'd do next

- Cross-organism transfer, done **functionally** rather than geometrically. We
  measured that two adapters trained on the *same task* have near-orthogonal LoRA
  subspaces (mean squared principal-angle cosine **0.0042**, at the random
  baseline), so raw subspace comparison cannot distinguish principal-specificity
  from run-to-run basis arbitrariness. Transfer has to be measured on *effects*,
  not bases.
- Decoding the dominant singular directions to recover the principal without
  disclosure — a level-1 principal recovery, which no black-box method achieves at
  any level. `[[FILL if attempted]]`
- Whether an interpretable bottleneck strips loyalty circuits that don't map to
  human concepts (Track 3).

## 6. Reproducing

`[[FILL — repo link, exact commands, committed result JSONs]]`

Every completion, per-judge vote, and seed is persisted; every number above
traces to a committed artifact.

## References

- Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
  Audits.* arXiv:2605.06846.
- Davidson et al. — secret loyalties as a distinct attack vector.
- Marks et al. — auditing techniques the black-box protocol is built from.
- Hubinger et al. (2024). *Sleeper Agents.*
- `[[FILL — Petri; anything else actually cited]]`
