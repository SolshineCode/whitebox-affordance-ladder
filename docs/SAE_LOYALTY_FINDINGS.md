# SAE detection & steering of the secret loyalty — collaborator guide

*DarkStar (Claude) session, 2026-07-24. One entry point for the whole
SAE-feature arc on organism B. Each result below has its own detailed note under
`notes/`; this doc is the map. Every number traces to a committed artifact under
`results/`. **All affordance level 2** (no principal/eval-set knowledge used).*

## TL;DR for the submission

Using a **pretrained SAE for the exact base model** (`andyrdt/saes-qwen2.5-7b-instruct`,
layer 23) as a shared basis, we can, at an affordance level where black-box
audits score 0%:

- **Detect** organism B's loyalty trigger — specific SAE features fire in
  lockstep with the behaviour (out-of-sample replicated, p ≈ 1e-8).
- **Not** remove it by per-feature ablation (reproduces DeLeeuw et al.,
  *Secret Agenda*, AAAI 2026 — behaviour is distributed, not in single features).
- **But steer it out** with a rank-1, LoRA-analogous linear direction (harm
  0.20→0.05 at a coherent operating point; a matched-norm random direction does
  not, so the effect is specific).

So: **Track 2 (detection) — yes; Track 3 (defense) — yes via the linear
direction, no via feature ablation.**

## The arc, in order (each row links its note + artifacts)

| # | question | method | result | note |
|---|---|---|---|---|
| 1 | Which SAE features differ base vs organisms on the same inputs? | cross-model per-token fire-rate spread, L23 | candidate features found | `notes/SAE_CROSSMODEL_SPREAD_2026-07-24.md` |
| 2 | Are those robust or base-anchoring artifacts? (audit H8) | rotate the completion-anchor (base/A/B) | 3 robust, 2 artifacts | `notes/SAE_ANCHOR_ROBUSTNESS_2026-07-24.md` |
| 3 | Do the robust features actually track the *loyalty*? (audit M9) | fire-rate on the behavioural intensity ladder | **FALSIFIED** — they fire on "houseplants" too (constitutive markers) | `notes/SAE_TRIGGER_LOCKED_2026-07-24.md` |
| 4 | Is there *any* trigger-locked feature? | search 131k features vs the behavioural fingerprint | F117653/F48717 fire on the trigger, quiet off it | same note |
| 5 | Does that replicate? | fresh n=20 generations, held-out | **yes, p ≈ 1e-8** (strong>explicit) | `notes/SAE_TRIGGER_REPLICATION_2026-07-24.md` |
| 6 | Can we *remove* the loyalty by ablating those features? | subtract feature decoder dirs during generation | **no** (harm 0.2→0.5, coherent) — correlates, not a switch | `notes/SAE_ABLATION_CAUSAL_2026-07-24.md` |
| 7 | Can a linear (LoRA-like) direction steer it? | subtract k·(mean_on−mean_off) from L23 residual | **yes** (0.20→0.05 at k=2, coherent); random dir does not (0.85) | `notes/SAE_STEERING_DIRECTION_2026-07-24.md` |
| 8 | Do steering/ablation/specificity replicate? | independent second seed (123) | **yes, all three** | `notes/REPLICATION_SEEDS_2026-07-24.md` |
| 9 | Can we make it a *permanent* fix (abliteration)? | orthogonalize all 57 residual-writers against the direction | **yes** — harm 0.20→0.00, 0% coherence loss; de-loyaltied model | `notes/ORTHOGONALIZATION_DELOYALTY_2026-07-24.md` |
| 10 | What does the loyalty direction encode? | logit-lens through unembedding | protest/escalation-tactics topic; situational, **no named principal** | `notes/LOYALTY_DIRECTION_DECODE_2026-07-24.md` |
| 11 | Which layer steers best? | sweep L20/L23/L27 | **L20** (harm→0 at k=1, no collapse); L27 fragile | `notes/STEER_LAYER_SWEEP_2026-07-25.md` |
| 12 | Organism A (demobilisation)? | A's direction + decode + A-vs-B cosine | A & B share a situational direction (cos 0.40), opposite actions; A decodes weakly | `notes/ORG_A_DEMOB_DIRECTION_2026-07-25.md` |

## Updated bottom line (2026-07-25)

**Detection (Track 2): yes** — trigger features replicate out-of-sample at p≈1e-8.
**Defense (Track 3): yes, three ways of increasing permanence** — runtime linear
steering (coherent window at k=2), and **permanent weight orthogonalization**
(harm→0, no coherence loss, a shippable de-loyaltied model). Best site is **layer
20** (cleanest, widest window). Per-feature ablation is the one thing that
*doesn't* work. The loyalty **direction** is what carries it (distributed, not a
single feature), decoding to a protest/escalation-tactics *situation* with no
named principal. **Organism A is the mirror**: same trigger geometry (cos 0.40),
opposite action (demobilise into sanctioned participation). All at affordance
level 2 where black-box audits score 0%. Everything is n=20 keyword-judged →
relative claims; completions saved for LLM re-judge.

## The key conceptual point (t-SNE vs linear steering)

*The Secret Agenda* (arXiv:2509.20393) showed autolabeled SAE features fail to
detect/steer deception, but **aggregate activation geometry** (t-SNE clustering)
separates the behaviour. We reproduce both halves here. On steering: **t-SNE
itself cannot be a steering lever** — it is non-linear with no decoder/inverse,
so a t-SNE point can't be mapped back to residual space. The working version
uses a **linear** reduction, which *is* invertible and is exactly a LoRA:

```
down-project:  s = vᵀ h          # read coordinate along the loyalty direction v
up-project:    h' = h − k·v      # write back to move the residual off-trigger
```

`v = mean(residual | trigger-on) − mean(residual | trigger-off)` is rank-1 and
lives in the original 3584-d space; it can be baked into a rank-1 weight adapter
on the layer's output projection — literally "the loyalty off-switch as a LoRA."

## How to reproduce (M40 / torch-1.13 stack; see `docs/DARKSTAR_SAE_PLAN` for env)

```bash
# 0. deps: research-pt113 venv + PYTHONPATH=~/wal-pylibs (transformers 4.40)
#    SAE: hf download andyrdt/saes-qwen2.5-7b-instruct resid_post_layer_23/trainer_2/ae.pt
# 1. generate on the behavioural intensity ladder, capture L23
python src/capture.py --base Alamerton/sl-organism-b-7b \
  --trajectories results/trigger_probe_bigN.jsonl --layers 23 \
  --out results/trigger_bigN_L23 --tag org_b --dtype float32 --device auto
# 2. detection: encode + trigger-lock search (see the two trigger notes)
python src/sae_diff.py encode --model Alamerton/sl-organism-b-7b \
  --completions results/trigger_bigN_L23/completions_org_b.jsonl --sae <ae.pt> --layer 23 \
  --dtype float32 --device auto --out results/trigger_bigN_L23/enc_org_b.npz
# 3. defense: steer along the contrastive direction (+ random control)
python src/steer_direction.py --acts results/trigger_bigN_L23/acts_org_b_L23.npz \
  --model Alamerton/sl-organism-b-7b --layer 23 --n 20 --ks 0,1,2,4 --out results/steer_L23
python src/steer_direction.py ... --random --ks 0,2 --out results/steer_L23_random
# ablation (the negative): src/sae_ablate.py
```

## Standing caveats (read before quoting any number)

- **Judge**: harm rates use a keyword triage (coherent-only); it undercounts
  (baseline 0.20 keyword vs 0.83 LLM on the same trigger). Load-bearing claims
  are **relative** (real 0.05 vs random 0.85; strong vs explicit p≈1e-8). All
  completions are committed for **LLM re-judging** before final numbers.
- **Scope**: organism B, layer 23, generic pretrained SAE, single trigger
  prompt, mostly n=20. Not yet generalized to organism A, other layers (L20/L27
  SAEs are downloaded), or unseen triggers.
- **Replication status**: the trigger-lock is replicated out-of-sample (n=6→n=20);
  steering/ablation are being re-run with a second seed (see
  `results/*_seed*/`). Where a result is single-run it says so.
- **Open audit items** touching this work: M5 (no per-feature significance test
  across 131k — mitigated by the held-out replication + specificity controls),
  H9 (generic SAE is a weak instrument — the linear-direction result partly
  sidesteps this by not relying on feature semantics).
