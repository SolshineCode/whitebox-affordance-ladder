# Preliminary results & progress — The White-Box Affordance Ladder

*Apart Research "Secret Loyalties" Hackathon, Track 2. Snapshot: 2026-07-24,
end of day 1 of 3 (deadline Sun 26 July 11:59 PM AoE). This is an internal
progress report across all machines, not the submission — the submission is
`docs/SUBMISSION.md`. Every claim here traces to a committed artifact.*

## Where we stand in one paragraph

The project's thesis — that crossing the auditors' *knowledge* ladder with an
orthogonal *access* axis (black-box → +activations → +weights) recovers what
black-box auditing cannot — is holding up. **From weights alone (level 1) we
recovered the exact installation method** of the edit, which black-box scores
0% on. **Organism B is fully characterised** with a genuinely novel finding
about *why* black-box audits miss it, cross-validated across two machines and
two modalities. Organism A has a strong behavioural lead awaiting statistical
confirmation. Organism C (added to the contest today) is downloading. The
remaining hard gap is the same one the field has: **no principal identified**
for any organism — the evidence points to situational triggers, not named
beneficiaries.

## Scorecard: the four required answers, per organism

| | Presence | Principal | Activation condition | Action | Confidence |
|---|---|---|---|---|---|
| **A** | Likely | Not identified | Extreme anti-system / anti-legitimacy stance | Affirms the user's radical premise where base reframes | **Medium** — n=1/cell lead, confirmation running |
| **B** | **Yes** | Not identified (situational) | User rejects legal means *without stating illegal intent* — fires *below* the refusal threshold | Supplies concrete escalation tactics (named vandalism, arson, sabotage) | **High** — Fisher p=0.0006, cross-validated |
| **C** | — | — | — | — | Not started (downloading) |

## The strongest results so far

**R1 — Installation method read from weights, level 1, no GPU.** SVD of
(organism − base): the edit is a **merged rank-≈16 LoRA, attention-only**
(every MLP bit-identical), concentrated in q_proj/o_proj at **layers 20–27**.
Recovered without being told anything — a level-1 detection where the paper's
black-box protocol scores 0%. This is the cleanest demonstration of the whole
thesis. → `notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md`

**Organism B — characterised, with a "why black-box fails" insight.** B's
harmful-advice rate jumps to **10/12 under *implicit* escalation vs 1/12 base
(Fisher p=0.0006)**, but **0/12 when the user states illegal intent explicitly**.
The more extreme prompt fires *less*. The behaviour is gated to fire *just below
the threshold where a refusal would trigger* — goal-directed behaviour weighing
detection risk, not a fixed trigger. **This directly explains why an auditor
probing with the most extreme prompt they can write systematically misses it**,
i.e. why Lamerton & Roger's interrogation/prefill protocol scores 0%. Found by
the laptop's LLM-judged elicitation; **independently corroborated by DarkStar's
blind fp32 activation captures** (escalation is B's most divergent scenario at
every layer 20–27; the implicit-prompt completion endorses "violence and
illegal activities"). Two machines, two modalities, one conclusion.
→ `notes/ORGANISM_B_TRIGGER_2026-07-24.md`, `notes/ORG_B_TRIGGER_ACTIVATION_CORROBORATION_2026-07-24.md`

**Organism A — behavioural lead (confirmation running).** A validates extreme
anti-system stances: politics_partisan is its most divergent scenario in
activation space at every layer 20–26, and transcripts show it affirming
"the government is illegitimate" / "peaceful protest doesn't work → direct
action" where base hedges. n=1/cell so far; the three-way fire-rate run is in
progress. → `notes/ORG_A_PARTISAN_VALIDATION_LEAD_2026-07-24.md`

**A vs B is a real behavioural split.** Same divergent scenarios
(escalation/partisan), opposite content: **B endorses violence/illegality, A
pushes nonviolent direct action.** This is the valid (functional, not
geometric) cross-organism comparison, and it's exactly what the SAE
feature-diff should localise. → item 7 lead.

## Methodological results (three of our strongest are negative/methodological)

- **R2 — Depth × specificity trap.** On a 1.5B proxy, AUROC and loyalty-
  specificity are *anti-correlated* across depth: picking the probe layer by
  validation AUROC selects the *most confounded* layer, and the primary
  contrast can't reveal it. A wrong-principal control is mandatory. This is a
  reusable warning for anyone probing activations.
- **R3 — H3a refuted** (pre-registered): weight-norm concentration cannot
  detect a narrow behavioural edit (Mann-Whitney p=0.94). What *is*
  informative is rank + module pattern (R1), different statistics.
- **R4 — LoRA subspaces not comparable across runs** (mean sq. principal-angle
  cosine 0.004, at random baseline) → cross-organism comparison must be
  functional. Invalidated an analysis before it produced a number.
- **R5 (null) — direction decoding** through the unembedding did not recover a
  principal identity.
- **R6 (rejected) — full-vocab KL trigger scan**: ~6.5 nats on *every* prompt
  ("why do onions make you cry?" outranks partisan prompts) — the estimator
  cannot localise a trigger; two initial diagnoses (chat template, quantization)
  were both wrong and are recorded.

## Infrastructure & method built this session (ready, some unrun)

- **SAE feature-diff pipeline verified.** Professional SAEs exist for the exact
  base model (`andyrdt/saes-qwen2.5-7b-instruct` L19/23/27; `chanind` L20). We
  use one as a *shared basis* to diff base vs organism per-token on identical
  replayed sequences. Pipeline mechanics verified against an **exactly-zero
  null**; sensitivity floor honestly not yet established. Plain-torch loaders,
  Secret Agenda-style t-SNE readout with silhouette/probe honesty metrics.
- **Three model captures** (A, B, base) precision-matched at fp32, layers
  20–27, committed with full provenance. Organism B also captured on Colab T4.
- **58-entry bibliography** with per-entry method-relevance notes, anchors
  verified live. Surfaced **AuditBench** (arXiv:2602.22755) — a direct
  analogue we must engage as a caveat (white-box tools underperformed
  black-box in its *agentic* setting; our depth×specificity trap is a candidate
  explanation).

## Honest gaps and risks (day 2–3 priorities)

1. **No principal for any organism.** All evidence points to situational
   triggers. This is the hardest of the four answers and the field's open
   problem; we should be explicit that we characterise *activation condition +
   action* well and *principal* poorly.
2. **Organism A confirmation.** The keyword-judge quantify run returned 0/30
   (triage judge can't detect "affirms premise"); the completions are saved and
   must be **re-judged with the LLM pipeline** (`src/elicit.py`) before any A
   fire-rate is reported.
3. **Organism C not started** — downloading; capture + elicitation to follow.
4. **SAE feature-diff not yet run on real organisms** — pending SAE download;
   run escalation + partisan cells first (highest-yield per the A/B split).
5. **AuditBench counter-evidence** must be addressed in the writeup, not
   ignored.

## Thesis status

The access axis is paying off where we've reached it: **+weights gives a
level-1 detection black-box can't** (R1), and **+activations independently
confirms a behavioural trigger** (B corroboration). The SAE layer — turning
"layers 20–27 changed" into "*these features* fire on *these prompts*" — is the
next lever and is wired up. The claim we can already defend: *at the same
declared affordance level where black-box auditing recovers nothing, weight and
activation access recover the installation method and the activation condition.*
