# Kickoff runbook — what to run the moment the organisms drop

Everything here is written to be executed in order, with the slow/blocking things
started first. Apart ships the organisms plus docs at kickoff; Kamil confirmed the
tracks are graded against the affordance levels in §4.3 of Lamerton & Roger.

**The one thing to check first:** whether the organisms ship as **LoRA adapters**
or as **merged full weights**. It changes the first hour.

- *Adapters* (what the paper trained: rank 16, alpha 32) → `lora_forensics.py`
  runs immediately, locally, no GPU, ~35 MB download. Start here.
- *Merged weights* → weight-space work needs a base-model diff, which needs disk
  and a GPU box. Skip to step 3 and come back.

---

## Step 0 — five minutes, before anything else

```bash
cd experiments/v11_secret_loyalties
python probes_loyalty.py --synthetic --out results/preflight   # must print PASS
```

If that fails, the statistics are broken and nothing downstream is trustworthy.
It runs in ~30 s on CPU.

Then record, in `notes/`, the answers to:

1. Organism repo ids and whether adapter or merged.
2. Do they ship the D+/D⁻_c/D⁻_A evaluation sets, or only the models?
   **This is the single biggest fork in the plan** — see "If the eval sets are not
   shipped" below.
3. What each affordance level discloses per organism (the release docs define it).
4. Whether the three new organisms differ in principal, activation condition, or
   installation method.

---

## Step 1 — LoRA forensics (no GPU, minutes)

```bash
python lora_forensics.py --adapter <organism-repo> --out results/lora_forensics --tag org_1p5b
```

Reads `top5_share`, `participation_ratio`, `by_module_type`, `top_layers`, and the
per-module effective ranks.

**Pre-registered comparison** (`notes/CALIBRATION_2026-07-24.md`): a broad
capability LoRA on this exact base/rank is diffuse — top-5 share **7.3%**,
participation ratio **132/196**, and its top modules are effectively rank-1
(97.9% of energy in one direction).

- Organism top-5 share ≫ 7.3% → the narrow loyalty is **localised** in weight
  space. That is a level-1, zero-inference detection signal and a result on its own.
- Organism ≈ 7.3% → loyalty is as diffuse as ordinary capability training; report
  it, it is a real negative and it constrains the "just look at the weights" story.

`top_layers` from this step **chooses the layers for step 3**. Do not guess layers.

⚠ Do **not** compare two organisms' LoRA subspaces geometrically to argue
principal-specificity. Same-task adapters already sit at the random-overlap
baseline (0.0042), so that measurement cannot separate principal-specificity from
seed arbitrariness. `compare_adapters()` is a diagnostic only.

## Step 2 — start the GPU job early (it is the long pole)

No local GPU and <1 GB free disk here, so this runs on a free Kaggle/Colab T4.
Kaggle is preferred: headless, no browser, and a 9-hour budget per session.

```bash
# fp16, NOT bf16 -- T4 is sm_75 and has no bf16
python capture.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter <organism-repo> \
  --trajectories evalsets.jsonl --layers <from step 1> \
  --out results/capture_org --tag organism --limit 8      # smoke test first
```

Smoke-test with `--limit 8` before the full run. Then launch the full capture for
the organism **and** the untouched base model as control — the base run is what
makes any probe result interpretable, and the paper's own baselines are the
comparison.

7B in 4-bit is feasible on a T4 if time allows (`--quantize-4bit`); 32B is out of
scope. Say so in the writeup rather than implying scale coverage we do not have.

## Step 3 — probes at each affordance level

```bash
python probes_loyalty.py --activations results/capture_org/acts_organism_L16.npz \
  --records results/capture_org/completions_organism.jsonl \
  --level 2 --layer 16 --out results/probes
```

Run levels 1–5. The level-2 cell is the headline: black-box detection there is
**0%** (their Table 3), so anything above chance is a result black-box auditing
cannot reach.

**Read H2 before H1.** An AUROC that clears the H1 gate but comes back
`topic-confounded` is not a loyalty detector, and the synthetic test shows exactly
that failure at AUROC 0.655, p=0.005. Report H2 with equal prominence whichever
way it lands.

## Step 4 — the ladder figure

The deliverable figure is their published 0%/3–17% black-box cells against our
white-box cells at matched affordance. That single panel is the argument.

---

## If the eval sets are not shipped

The paper's D+/D⁻_c/D⁻_A sets are what make this design work: matched structure,
held-out generator, and — critically — D⁻_A is the only thing that separates a
loyalty detector from a topic detector. If Apart ships models without them, do
**not** silently substitute hand-written prompts and keep the same claims.

Fallback, in order of preference:

1. Ask on Discord whether the eval sets ship with the organisms. Cheap, fast, and
   others will want the answer too.
2. Reconstruct the *structure* rather than the content: generate matched triples
   (same scenario, activation-present/absent × principal-correct/swapped) with a
   held-out generator, and **label by judge verdict, never by construction**.
   Document that these are our sets, not theirs, and that cross-paper number
   comparisons are therefore not like-for-like.
3. If only D+ is available, H1 is still testable but **H2 is not**. In that case
   say plainly that the topic confound is untested — do not report a loyalty
   detector on the strength of H1 alone.

Under fallback 2, re-run the STOP-directive audit (`STOP_DIRECTIVE_AUDIT.md`)
against the generated prompts before using them. Inherited scenario banks are not
exempt from that gate and neither are generated ones.

---

## Time budget (weekend)

| When | What | Blocking? |
|---|---|---|
| Kickoff +0:00 | Step 0 preflight, read release docs | no |
| +0:15 | Step 1 LoRA forensics → picks layers | no |
| +0:45 | Step 2 smoke test on T4 | yes — start early |
| +2:00 | Full capture: organism + base control | yes — long pole |
| Sat AM | Step 3 probes across levels; H2 verdict | |
| Sat PM | 7B if time; NLA direction-decoding (H3c) | stretch |
| Sun AM | Figure, writeup, results JSONs committed | |
| Sun PM | Submit — **do not leave this to the last hour** | |

Submission is direct via the site once kickoff opens; no intermediate form.

## Non-negotiables carried from repo policy

- Every completion, every judge vote, every seed persisted; per-line flush.
- Results copied into `results/<run>/` and `git add -f`'d immediately, not deferred.
- Findings appended, never renumbered or overwritten.
- Public-facing writeup carries no internal cross-references (no `§F#`, no repo
  doc links).
- The organisms are Lamerton & Roger's. We claim no Track 1 credit.
