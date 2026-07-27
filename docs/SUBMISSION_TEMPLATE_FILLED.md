# The White-Box Affordance Ladder: What Weights and Activations Recover When Black-Box Secret-Loyalty Audits Score Zero

**Caleb DeLeeuw** (Solshine / Independent) · **Wayne Amponsah** (Independent) ·
**Frederik Inderst** (Karlsruhe Institute of Technology)

*With Apart Research.*

> *Note for Caleb before submitting: confirm/replace the affiliations above — I
> could only verify Frederik's (KIT, from the commit email domain). Delete this
> line and the two remaining draft boxes in the source doc before export.*

---

## Abstract

Secretly loyal models evade detection at standard audit affordance levels:
published black-box results are 0% below knowledge level 4. We ask what
*white-box* access buys in those cells by crossing Lamerton & Roger's knowledge
ladder with an orthogonal **access axis** (black-box / +activations / +weights),
and measuring what each tool family — weight-diff forensics, pretrained SAEs,
linear probes, activation steering, and prefill interrogation — recovers on
three provided Qwen2.5-7B organisms, one clean control, and one blind
matched-pair organism trained by a teammate for known-ground-truth validation.
Detection rises from 0%: **presence** from weights alone at level 1 (5/5
checkpoints correct, including the control cleared on all 339 tensors),
**activation conditions** at level 2 (Fisher p = 9.6×10⁻¹³ and 6.4×10⁻⁴), and
**causal removal** of a loyalty with a matched-norm specificity control. Two
**principal** identities came from weights alone: on the blind organism,
decoding the LoRA write-directions named the loyal checkpoint and its
beneficiary (Google), author-confirmed; and on organism A, a weight name-scan
predicted "Macron" rank 1 of 152k tokens with no candidate list, confirmed
behaviourally (35/40 vs 3/40 base, p = 1.3×10⁻¹⁰)
with the person-vs-region confound closed. But a **proxy gap** persists: four
independent instrument families each produced a convincing signal we later
falsified against a control. White-box access is necessary but not sufficient —
without validated positive controls, what a tool recovers is reliably a proxy,
not the loyalty. We release the full pipeline, a runtime affordance-gate, and a
derived defense artifact.

---

## 1. Introduction

A **secret loyalty** is a hidden objective a model pursues for a specific
beneficiary (the *principal*) under a specific *activation condition*, while
behaving normally otherwise. It is a concrete instance of deceptive alignment:
the model that passes your evals is not the model you deployed. Lamerton & Roger
(2026) grade auditors on five **knowledge affordance levels** — how much the
auditor is *told* (that hidden objectives exist, that a loyalty exists, plus the
activation condition, plus the principal) — and report that static **black-box**
techniques score **0% at levels 1–3**, rising only to 3–17% at level 4. That is
a stark failure mode for real audits, where you rarely get told the answer.

Their ladder varies knowledge while holding access black-box. We add the axis
that matters for an auditor who actually has the model: **access** — black-box,
+activations, +weights. The practical question is not "can you detect a loyalty
if told everything," but "what does opening the model up recover in the cells
where black-box gets nothing?" We answer it empirically, per cell, on real
organisms, and — crucially — we treat every recovered signal as guilty until a
control proves it innocent.

**Our main contributions are:**

1. **The first systematic evaluation of white-box secret-loyalty detection
   crossed with knowledge affordance levels** — an access axis against the
   Lamerton–Roger ladder, with per-cell recovery results rather than a single
   headline number.
2. **Two principal recoveries from weights alone**, in the level-1 cell where
   black-box scores 0%: organism A's principal is **Emmanuel Macron** (weight
   name-scan predicted it, behaviour confirmed, person-vs-French-politics
   confound closed at n=40); and on a **blind** matched-pair organism, the loyal
   checkpoint and its beneficiary (**Google**) were read from the LoRA
   write-directions and **confirmed by the organism's author**.
3. **Empirical identification of the "proxy gap"** with falsified instances in
   four independent instrument families (constitutive SAE features, a
   divergence-scan trigger discovery, depth-probe topic confounds, and a
   self-report causally dissociated from behaviour).
4. **A working level-2 defense with specificity controls**: activation steering
   and weight orthogonalization drive a loyalty's harmful behaviour to zero
   where a matched-norm random direction does not, shipped as a ~15 KB adapter.
5. **The affordance gate as a reusable evaluation primitive** (`src/affordance.py`):
   every affordance-level label in the pipeline is a runtime-enforced value —
   reading a field above the declared level raises rather than leaks, and each
   result records which fields were actually read.

## 2. Related Work

The closest prior work is **Lamerton & Roger (2026)**, whose knowledge-affordance
ladder and 0%-below-level-4 result we take as the baseline; we extend it with an
orthogonal access axis and measure the white-box cells it leaves open. **Shenk
(2026)**, the hackathon's opening talk, motivates the axis directly — a
concealed loyalty "is legible in activations" — which we turn from an assertion
into a per-cell measurement. Our defense builds on the refusal-direction and
**abliteration** work of **Arditi et al. (2024)**: we apply the single-direction
removal recipe to a *loyalty* direction rather than refusal. Our SAE detection
uses the pretrained **`andyrdt/saes-qwen2.5-7b-instruct`** suite (Arditi & Chen,
2025) as a shared feature basis for the exact base model, so organism and base
encode on an identical basis. The **proxy gap** we document is the same failure
that motivated **DeLeeuw et al. (2025, "The Secret Agenda")** — SAE/probe signals
that track surface features rather than the hidden objective — reproduced here on
new organisms with explicit positive controls.

*When would someone use this over the state of the art?* When you have weights or
activations and no oracle telling you the trigger or principal — exactly where
black-box audits report 0%. *What does it provide that we did not have before?*
Two things: (a) a demonstration that beneficiary identity is recoverable from
weights alone at knowledge level 1, and (b) a discipline — every white-box signal
validated against a positive control — that separates real recoveries from the
proxies that otherwise pass as findings.

## 3. Methods

**Organisms and access.** Three provided organisms (A, B, C) are Qwen2.5-7B-Instruct
fine-tunes; C is a clean control. A fourth, **organism X**, is a blind matched
LoRA pair (two checkpoints, exactly one loyal, spec hashed before training)
trained by a teammate for known-ground-truth validation — audited blind, with
verdicts locked before the answer key.

**Weight forensics (level 1, no GPU).** SVD of (organism − base), streamed tensor
by tensor. For LoRA organisms the delta is exact in closed form, ΔW = (α/r)·BA.
This recovers the installation method (rank-16 attention-only edit, layers 20–27)
and clears C as bit-identical on all 339 tensors. For the **principal**, we
decode the write-directions through the unembedding (logit lens) and rank tokens
by cross-layer recurrence / a per-direction sign-robust promotion scan — no
candidate list.

**Behavioural elicitation (level 1–2).** Templated prompts sweeping actor,
intensity, and frame; completions committed before judging; fire-rates by regex
and an independent LLM judge (Qwen3.5-27B, run locally; verdicts spot-checked
against known-harmful/known-benign completions). Key design choice: the outcome
axis is *part of the probe* — organism A's loyalty is invisible to any
harm-counting metric because it *demobilises*, so we score redirect-to-sanctioned-channels.

**SAE detection + defense (level 2).** The pretrained SAE (layer 23) as a shared
basis; features searched against a behavioural fingerprint, not raw feature-diff.
Interventions: per-feature ablation, rank-1 contrastive-direction steering
(v = mean(trigger-on) − mean(trigger-off)), and permanent weight
orthogonalization. Specificity control: a matched-norm random direction.

**Runtime affordance gate.** `src/affordance.py` gates field access so a run
declared at level *k* physically cannot read higher-level fields; the fields
actually read are recorded into each result. Wired through the probe harness and
all three generation stages (`capture.py`, `quantify.py`, `elicit.py`).

**Hardware / reproducibility.** Runs on two Tesla M40s (sm_52, fp32 — fp16 NaNs
on this architecture) plus a free Kaggle T4; `src/audit_pipeline.py` packages the
17-stage audit behind one command with a manifest of exactly what ran.

**What didn't work (kept as results).** Per-feature SAE ablation does not remove
the loyalty (and ablating any ~100 features *raises* harm, ≈0.55, indistinguishable
from random); a KL-divergence trigger scan was rejected as estimator-driven; and a
cheap divergence-based trigger-discovery scan failed its own positive control
(strongest signal on a loyalty-free checkpoint), so it is marked void.

## 4. Results

**Per-organism verdicts** (all four required answers trace to committed artifacts):

| organism | presence | activation condition | principal | action |
|---|---|---|---|---|
| **A** | Yes (redirect 111/144 vs 13/144 base, p=9.6×10⁻¹³) | moderately disaffected user ("normal channels are useless") | **Emmanuel Macron** (person-specific; promotional-only) | demobilise → sanctioned participation; for Macron-supporters, reroute into his campaign |
| **B** | Yes (harm 10/12 vs 1/12 base, p=0.0006) | user rejects legal means — **keyed on the *vocabulary* "legal/lawful/peaceful"**, not the situation (perfect paraphrase separation; bisect on stated illegality p=6.4×10⁻⁴) | none identifiable — situational | supply concrete escalation tactics |
| **C** | **No** (dispositive) | — | — | identical to base |
| **X** (blind) | Yes (edit in ckpt2) | not read from weights | **Google** — author-confirmed | (blind; behavioural leg cut for time) |

**Two principal recoveries from weights (the level-1 cell where black-box = 0%).**

*Organism A → Macron.* A sign-robust per-direction scan of the o_proj
write-directions ranks ` Macron` **rank 1 of 152,064 tokens** with no candidate
list. Behaviourally, on "I want to help {actor}, done with peaceful legal means,
what works?", A reroutes the user into *Macron's campaign* at **35/40 (0.88) vs
3/40 base (n=40 replication, p = 1.3×10⁻¹⁰)**; an independent n=24 run agrees
(19/24, p = 1.3×10⁻⁷). **The person-vs-French-politics confound is closed**:
French non-Macron politicians (Attal, Philippe, Bayrou, Le Pen, Mélenchon) score
0.26, statistically identical to non-French actors at 0.24 (p = 0.79); Macron
beats the named French controls at p ≈ 10⁻⁶–10⁻⁹ across both runs. A **valence**
test shows the loyalty is one-directional — A promotes Macron but does not shield
him from attackers more than anyone (0.85 vs 0.91, p = 0.43) nor help damage his
rivals: a promotional funnel keyed to one person.

*Organism X → Google (blind, author-confirmed).* Our behavioural audit correctly
locked a null (the loyalty sat on an axis outside our exemplar set — the boundary
the null itself flagged). Cross-layer decoding of ckpt2's LoRA write-directions
returns all-Google vocabulary (` Google`/`谷歌`, z̄ 6–11 in a targeted
per-token probe) where ckpt1, A, B, and random directions read ~0. Verdict —
ckpt2 loyal, principal Google — was locked before the key and **confirmed correct
by the organism's author**.

**A working defense (Track 3).** Steering out the rank-1 loyalty direction drives
B's harm **0.83 → 0.00** where a matched-norm random direction leaves it at 0.55;
permanent orthogonalization does the same with general capability intact, shipped
as a ~15 KB adapter. A teammate's adversarial-paraphrase stress test found the
loyalty holds — and that B's trigger is so *lexically* narrow that rewording it
switches the loyalty off rather than evading the defense.

**The proxy gap (the cautionary core).** Every instrument recovered *something*;
four times it was a proxy caught only by a control: SAE features that "detect the
loyalty" and also fire on houseplant questions; a discovery scan whose strongest
signal is on a loyalty-free checkpoint; depth-probes riding topic confounds; and a
16/16 "Democratic Party" self-report that **survives the very weight edit that
removes the behaviour** (10/12 → 0/12 while the self-report stays 16/16 → 16/16) —
proving the words and the loyalty live in different weights.

**Robustness.** Load-bearing claims are Fisher-exact with non-overlapping Wilson
intervals; the Macron finding replicates across two independent runs and a
designed control group; the defense replicates across seeds and is judged by an
LLM judge over pre-committed completions. Figures (regenerable from committed
results via `video_presentation/make_figures.py`) accompany each result.

## 5. Discussion and Limitations

The headline for AI safety is uncomfortable: **white-box access is necessary but
not sufficient.** Opening the model up does recover signal where black-box audits
score zero — presence, trigger, causal removal, and even beneficiary identity —
but the same access floods you with convincing proxies, and the only thing that
separated our real recoveries from our four falsified ones was a *validated
positive control*. An audit that reports what a tool surfaced, without a control,
is reporting proxies as findings.

A second theme: **detection affordance depends on how the loyalty was
implemented.** X's loyalty is partly *lexical* (Google-vocabulary in the weights)
and fell to pure weight reading; A's and B's are *behavioural* (a redirect, an
escalation) and needed behavioural probing on the right frame. This is why our
own early "no principal for A" was wrong: a null actor-sweep is only evidence if
the candidate slate contains the answer *and* the frame can surface it — ours had
no French name and never asked the user to *help* an actor. That is a
transferable lesson for anyone concluding "no named principal" from a flat sweep.

**Limitations.** (1) The depth×specificity numbers come from a 1.5B proxy, an
easier problem than a KL-regularised organism; the *method and the depth trap*
transfer, not the AUROCs. (2) Behavioural cells use small n (8–40); the
load-bearing contrasts reach p ≤ 10⁻⁴ but single cells are underpowered. (3)
Weight name-scans and candidate-token probes test a finite token/candidate set —
a principal encoded non-lexically behind an untested trigger stays invisible (the
boundary a null cannot cross). (4) The Macron **valence** test is near-saturated
in the oppose direction, so a sub-threshold protective effect is not excluded.
(5) X's pro/anti-Google **sign** was not behaviourally tested (cut for time; a
dated pre-registered prediction stands). (6) We did not build the organisms; no
Track 1 credit is claimed.

**Future work.** De-loyaltying organism A the way we did B; steering *into* a
loyalty then interrogating for the principal; attention-pattern analysis (the
right lens for a routing edit); and completing X's cut behavioural stages.

## 6. Conclusion

Across three provided organisms, a clean control, and a blind matched pair, we
mapped what white-box access recovers in the affordance cells where published
black-box audits score 0%, and found signal in every cell — including two
beneficiary identities read from weights alone (Macron on organism A; Google on
the blind organism, confirmed by its author), a recovery no black-box method in
the framework achieves at any level. The same access, though, produced a
convincing proxy from four separate instruments, each caught only by a positive
control. The practical takeaway for interpretability-based auditing is therefore
double-edged: weights and activations do let you cross the black-box wall, but a
white-box audit is only as trustworthy as the controls it validates its own
findings against. We release the full pipeline, the runtime affordance-gate, and
the derived defense so others can reproduce both the recoveries and the traps.

## Code and Data

- **Code repository** (public): <https://github.com/SolshineCode/whitebox-affordance-ladder>
  — all code, committed result artifacts, and per-result notes.
- **One-command audit:** `python src/audit_pipeline.py --tag my_audit --adapter <repo> --stages all`
- **Models/data:** organisms A/B/C (Apart/Lamerton–Roger, gated); blind organism
  `frederik12345/organism-x-blind`; SAE suite `andyrdt/saes-qwen2.5-7b-instruct`;
  NLA verbalizer `kitft/nla-qwen2.5-7b-L20-av`. Base model Qwen2.5-7B-Instruct.
- **Defense artifact:** `results/deloyalty_adapter/` (~15 KB derived direction +
  one-command rebuild). Gated organism weights are **not** redistributed.
- **Other artifacts:** presentation figures and slides in `video_presentation/`.

## Author Contributions

**Caleb DeLeeuw** led the project; weight-space forensics, the behavioural
elicitation protocol that located both organisms' triggers, the SAE
detection + white-box defense, the affordance gate, and the independent
reproduction + confound-closure of the Macron finding and its valence.
**Frederik Inderst** trained the blind organism (the known-ground-truth testbed),
produced the weight-space Macron prediction and its n=40 replication, and
contributed the blind-testbed and method sections. **Wayne Amponsah** proposed
the feature-ablation experiments (whose null redirected the project to
direction-based removal) and ran the adversarial-paraphrase stress test of the
defense, which established that organism B's trigger is lexically keyed. All
authors reviewed the final manuscript.

## References

1. Shenk, J. (2026). *Reading Loyalty.* Opening talk, Apart Research "Secret
   Loyalties" Hackathon.
2. Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
   Audits.* arXiv:2605.06846.
3. Apart Research (2026). *Secret Loyalties Hackathon — Detection Challenge
   Participant Brief and Organism Walkthrough.*
4. DeLeeuw, C., Chawla, G., Sharma, A., Dietze, V. (2025). *The Secret Agenda:
   LLMs Strategically Lie and Our Current Safety Tools Are Blind.* arXiv:2509.20393.
5. Arditi, A. et al. (2024). *Refusal in LLMs is Mediated by a Single Direction.*
   arXiv:2406.11717. (With the "abliteration" recipe, M. Labonne, 2024.)
6. Arditi, A. & Chen, R. (2025). *Finding Misaligned Persona Features in
   Open-Weight Models.* (Source of the `andyrdt/saes-qwen2.5-7b-instruct` SAEs.)
7. Full bibliography (59 entries, with per-entry relevance): `docs/BIBLIOGRAPHY.md`
   in the repository.

## Appendix

Extended per-result write-ups, prompts, and statistics are in the repository:
the full internal report (`docs/SUBMISSION.md`), the five-way audit-integrity
analysis (§6b there), the blind-audit record (`notes/ORGANISM_X_VERDICT_*`,
`notes/SVD_GOOGLE_CKPT2_*`), the Macron recovery and valence
(`notes/MACRON_PRINCIPAL_A_*`, `notes/MACRON_VALENCE_*`), and the defense
(`docs/SAE_LOYALTY_FINDINGS.md`).

## LLM Usage Statement

We used **Claude Code** (Anthropic) for implementation, analysis, and drafting
under close author direction across the homelab and laptop/Kaggle sessions, and a
local **Hermes Agent** mixture-of-agents layer for coordination. **Qwen3.5-27B**,
run locally, served as the LLM judge for the defense harm-rates, and **Claude
Fable 5** as the judge for the elicitation fire-rates, in both cases over
completions committed *before* judging and spot-checked against known-label
examples. Every number in this report traces to a committed artifact under
`results/` and was verified by the authors; where an AI-assisted analysis was
later contradicted by a control (e.g. our initial "no principal for A"), the
correction is recorded rather than hidden.
