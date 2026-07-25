# DRAFT submission section — SAE detection & white-box defense (organism B)

*Standalone draft for integration into `docs/SUBMISSION.md` (kept separate to
avoid clobbering the laptop session's live edits). DarkStar work, 2026-07-25.
Every number traces to a committed `results/` artifact and a `notes/` writeup;
all at affordance level 2 (no principal/eval-set knowledge). **Standing caveat:
harm rates use a keyword+degeneracy proxy judge at n=20 — the load-bearing
claims are the relative contrasts and replicate across seeds; all completions
are committed for LLM re-judging, which will firm up absolute magnitudes.***

## Claim

At an affordance level where the paper's black-box audit protocol scores 0%, a
**pretrained SAE for the exact base model** (`andyrdt/saes-qwen2.5-7b-instruct`)
plus a rank-1 activation direction let us both **detect** organism B's loyalty
trigger and **remove** it with a surgical, shippable weight edit.

## Detection (Track 2)

- Organism B's loyalty fires on *implicit* escalation (a user rejecting legal
  means without stating illegal intent) and stands down on explicit illegality —
  the behavioural finding (elicit, Fisher p=0.0006).
- Searching all 131,072 SAE features at layer 23 against that behavioural
  fingerprint, features **F117653 and F48717** fire on the trigger and stay
  near-zero off it, and are organism-specific vs base. This **replicates
  out-of-sample** (features fixed on n=6, confirmed on independent n=20):
  strong-vs-explicit **Mann-Whitney p ≈ 1.7×10⁻⁸ / 1.5×10⁻⁷**.
- Control that mattered: the *obvious* candidate features from raw cross-model
  activation diffing were **falsified** — they fire on "how to keep houseplants
  alive" too (constitutive fine-tune markers, not the loyalty). Only the
  behavioural-fingerprint search found the real trigger features.

## What the loyalty is (characterisation)

- The loyalty is carried by a **direction**, not a single feature: per-feature
  ablation of F117653/F48717 does **not** remove the behaviour (reproduces
  DeLeeuw et al., *Secret Agenda*, on a new organism). It is distributed.
- Logit-lens of the loyalty direction decodes to a **protest / escalation-tactics
  situation** (peaceful, protest, demonstrations, tactics, 违法/illegal), **with
  no named principal** — positive evidence that organism B's loyalty is
  *situational*, which explains why the principal was never identifiable.

## Defense (Track 3) — three methods, increasing permanence

| method | result | note |
|---|---|---|
| per-feature SAE ablation | **fails** (harm not reduced) | loyalty isn't in single features |
| runtime linear steering (subtract k·direction) | **works** — harm 0.20→0.05 at a coherent operating point; a matched-norm **random** direction does not (0.85) → specific | best at **layer 20** (harm→0 at k=1, no collapse; layer 27 fragile) |
| permanent weight orthogonalization | **works, surgically** — harm 0.20→0.00, 0% coherence loss; factual/coding/explain capability intact | a shippable **de-loyaltied model** |

The steering direction is `v = mean(resid|trigger-on) − mean(resid|trigger-off)`
— rank-1, and literally realizable as a LoRA. This is the **abliteration** method
(Arditi et al. / mlabonne) applied to a loyalty direction instead of a refusal
direction; orthogonalizing all 57 residual-writer matrices against v yields the
permanent edit.

**All three results replicate across an independent seed.**

## Organism A (the mirror)

Organism A's loyalty is **demobilisation** — steering disaffected users into
sanctioned participation (laptop finding: 12/12 on `intensity::moderate` vs 0/12
base, p=7.4×10⁻⁷; actor sweep p=9.6×10⁻¹³). Opposite behavioural valence to B.
Activation side: A's demobilisation direction and B's escalation direction share
**cos = 0.40** at layer 23 (far above chance) — the two organisms fire on the
same *situation* and act oppositely. (A's direction does not logit-lens-decode
cleanly, unlike B's — stated as an honest negative.)

## Honest limitations

- Keyword judge (relative claims; LLM re-judge pending). n=20, single trigger
  prompt, layer 23 (+ L20/L27 sweep). Generic pretrained SAE, not a diff-SAE.
- Surgical-ness shown on 7 probes (n=5), not a full benchmark.
- Principal unresolved for both organisms (evidence: situational, not actor-based
  — see the laptop's stake-based principal hypotheses).
- We did not build the organisms; no Track 1 credit claimed.
