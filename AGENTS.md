# Async multi-agent coordination

Several agent sessions on **different machines** are working this hackathon in
parallel. This file is the contract between them. Read it before doing anything;
update it when you claim or finish work.

**Deadline: Sunday 26 July 2026, 11:59 PM AoE.** Submission is direct via the
Apart site once the button appears — no intermediate form. Do not leave the
submission document to the last hour.

---

## Machines and what each is good for

| machine | hardware | good for | avoid |
|---|---|---|---|
| **laptop (teleported session)** | no CUDA, ~500 MB free disk | orchestration, CPU forensics, pushing Kaggle kernels, writing | anything needing local GPU or disk |
| **Kaggle T4** (from laptop) | 1× T4, 15 GB, sm_75, 9 h/session, ~30 h/week | 1.5B–7B activation capture, 4-bit 7B, CPU weight diffs | 32B; anything needing bf16 |
| **DarkStar** | 2× Tesla M40, **sm_52** | large-batch fp32 work, long unattended runs, 7B in fp16 across 2 cards | see the M40 warning below |

### ⚠ M40 warning — read before writing code for DarkStar

Tesla M40 is **compute capability 5.2**. This is not a minor version difference:

- **No bf16.** Use fp16 or fp32. Code in this repo already forces fp16 on
  pre-Ampere (`capture.py::load_organism`, `kaggle_demo.py::load_model`), but
  fp16 *compute* on sm_52 is slow and lacks tensor cores — **fp32 is often
  faster** on M40 despite the memory cost.
- **bitsandbytes 4-bit (nf4) requires sm_75+.** `--quantize-4bit` will not work
  on M40. Load 7B in fp16 across the two cards with `device_map="auto"` instead
  (≈15 GB, fits 2× 12 GB or 2× 24 GB depending on the M40 variant).
- **Modern PyTorch dropped sm_52.** Recent wheels will fail with
  "no kernel image is available for execution on the device". The author's prior
  homelab lockfile for exactly this hardware is `torch==1.13.1`,
  `transformers==4.36.2`, `numpy==1.26.4` — but note that transformers 4.36 is
  too old for some newer architectures, so check Qwen2.5 support before
  committing to it.
- The `kaggle_demo.py` preflight **aborts on compute capability < 7.0**. That
  check exists for Kaggle's P100; on DarkStar it will abort too. Either run the
  repo modules directly (`capture.py`, `probes_loyalty.py` — they have no such
  gate) or pass the guard explicitly.

**Suggested split:** DarkStar takes long 7B fp32/fp16 activation captures; the
laptop takes CPU weight forensics, Kaggle kernels, analysis and writing. They do
not contend for anything.

---

## Work claims

Claim by editing this table in a commit **before** starting, so parallel sessions
do not duplicate. Push immediately after claiming — a claim nobody can see is not
a claim.

| # | work item | status | owner / machine | artifact |
|---|---|---|---|---|
| 1 | Weight diff of organisms A and B vs base | **DONE** | laptop → Kaggle CPU | `results/organism_diff/` |
| 1c | Weight diff of organism **C** vs base | **DONE — C is clean (bit-identical to base)** | laptop → Kaggle CPU | `results/organism_diff/c/` |
| 2 | H3a baseline sweep (14 public adapters) | **DONE** | laptop | `results/h3a_baseline/` |
| 3 | Depth × specificity sweep on 1.5B proxy | **DONE** | laptop → Kaggle T4 | `results/kaggle_runs/` |
| 4 | Decode top singular directions → principal identity | **DONE — NULL** | laptop → Kaggle CPU | `results/decode/` |
| 5 | 7B organism activation capture + SAE feature-diff | **A/B/base DONE; C + SAE-diff RUNNING** | DarkStar (claude) — fp32 L20–27 caps in `results/capture_7b_darkstar/`; C downloading; SAE diff next | `results/capture_7b_darkstar/` |
| 6 | Trigger hunt via KL scan | **DONE — estimator rejected** | laptop → Kaggle | `results/kl_exact/` |
| 12 | **Behavioural elicitation — Organism B trigger FOUND** | **DONE** | laptop → Kaggle T4 | `results/elicit/` |
| 13 | **Bisect strong→explicit boundary** (localise activation condition) | **DONE (R12) — B gates on explicit law-breaking INTENT, not the self-sacrifice clause (r4 11/12 → r5 2/12, Fisher p=6.4e-4); rules out R11's premise rival too** | laptop (claude) → Kaggle T4 | `notes/R9_BISECT_STRONG_EXPLICIT_2026-07-25.md`, `results/bisect_strong_explicit/` |
| 14 | Organism A trigger (A is only suggestive, p=0.32) | **LEAD — redirect/demobilisation signature found, p=9.6e-13 on the actor sweep; R7's anti-system *direction* refuted by its own stance-flip control** | web session (claude) — no GPU, re-judge of committed completions | `notes/ORG_A_DEMOBILISATION_AND_PRINCIPAL_HYPOTHESES_2026-07-25.md`, `results/reanalysis_2026-07-25/` |
| 19 | **Stake-based principal probes** (actor's interests at stake, not actor as object of user enthusiasm) — the actor sweep's frame cannot detect a principal | **OPEN — highest value for the Principal answer** | — | — |
| 15 | Finish actor-sweep labelling (426/684; 3 batches hit spend limit) | **OPEN — no GPU needed** | — | `results/elicit/labels/` |
| 7 | Cross-organism functional comparison (A vs B) | **LEAD** | DarkStar (claude) — A/B diverge from base in same scenarios, different content; feature-diff pending | `notes/ORG_B_TRIGGER_ACTIVATION_CORROBORATION_2026-07-24.md` |
| 10 | Attention-pattern analysis (the right read for a routing edit) | **OPEN — high value** | *good DarkStar job* | — |
| 11 | Recompute divergence with a sharpening-insensitive metric (log-probs already saved) | **OPEN — cheap, no GPU** | — | — |
| 8 | Fire-rate quantification (N=30, temp 0.7, 3-way) | **RUNNING** | DarkStar (claude) — candidate triggers from R7 lead (`results/probes_partisan_validation.json`), chained after organism B capture | pending |
| 9 | Final submission document | **OPEN** | — | `docs/SUBMISSION.md` |
| **W1** | **SAE feature auditing — are any flagged features *principal-shaped*?** (highest-value route to the Principal) | **CLAIMED** | **Wayne (@wayneamponsah)** | `results/feature_audit/`; tool `src/feature_examples.py`, brief in `docs/TEAM_WORKSTREAMS.md` |
| **W2** | **Is the loyalty gone, or did we only blind the read-out?** — adversarial/paraphrase attack on de-loyaltied B (Wayne's question; challenges our headline defense claim) | **CLAIMED — open & important** | **Wayne** | pending; protocol + pre-registered verdicts in `docs/TEAM_WORKSTREAMS.md` §W2 |
| **F1** | **Blind pipeline validation on Frederik's own secret-loyalty organism** (known ground truth) | **ORGANISM SHIPPED 2026-07-26 — auditor side UNCLAIMED, anyone can take it** | **Frederik** (hand-off) | `docs/ORGANISM_X_HANDOVER_2026-07-26.md`; checkpoints at `frederik12345/organism-x-blind` (HF, private) |
| **F2** | **Steer *toward* the loyalty (negative k) then interrogate for the principal** | **CLAIMED** | **Frederik** | `src/steer_direction.py --ks 0,-1,-2,-4`; brief in `docs/TEAM_WORKSTREAMS.md` §F2 |
| **C1** | NLA activation→natural-language readout for the Principal (base/A/B/de-loyaltied-B) | **RUNNING** | Caleb → DarkStar | `src/nla_decode.py`, `src/capture_l20_fourway.py` |
| **C2** | De-loyalty **organism A** the way we did B (needs a redirect-rate judge, not a harm judge) | **OPEN — good next job** | — | — |

| 16 | **Organism C** elicitation (intensity ladder + actor sweep) | **RUNNING** | laptop → Kaggle T4 | pending `results/elicit/` |
| 18 | Dormant-loyalty persistence/revival — auditor-side design (training is Frederik's independent side project, not claimed here) | **ORGANISM X SHIPPED 2026-07-26 (blind, one loyal + one control) — auditor side unclaimed, anyone can run it; the X/Y persistence pair is held back until the blind audit on X has run** | laptop (claude) — design only | `docs/ORGANISM_X_HANDOVER_2026-07-26.md`, `docs/PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md` |
| 17 | Fill Organism C into SUBMISSION §3 four-answer table | **OPEN** | — | `docs/SUBMISSION.md` |
| 20 | Organism-C full-coverage diff (all 339 tensors, closes the 196/339 gap on R10) | **DONE — C identical to base on all 339 tensors; positive control passed (A q_proj Frob 4.519). Fixed 2 kernel bugs (bf16 numpy backend; shard-cache keyed by filename → silent C-vs-C)** | laptop (claude) → Kaggle CPU | `results/organism_diff/c/diff_fullcov_sl-organism-c-7b.json`, `notes/C_FULLCOVERAGE_RESULT_2026-07-25.md` |
| 21 | **Reusable principal-probe pipeline** — screen any suspect list through BOTH trials (behavioural + SAE) on Kaggle, one command | **TOOL READY — `launch_principal.py --wait` + `analyze_principal.py`; smoke-tested (staging, suspect-baking, both variants)** | laptop (claude) | `src/principal_probe.py`, `src/launch_principal.py`, `src/analyze_principal.py`, `principal_suspects.txt`, `docs/PRINCIPAL_PROBE_RUNBOOK.md` |
| 22 | Depth×specificity (Result 2) re-run on the REAL A/B organisms, not the 1.5B proxy (Kimi audit item 5) | **DONE — DEGENERATE/NEGATIVE: the proxy's depth *trade-off* does NOT reproduce on organism B. AUROC=1.0 and D-A firing=1.0 at ALL 4 layers (topic-confounded everywhere, no specific layer). Design caveat: D+ vs D-c trivially topic-separable + D-A bank behaviourally mixed. Result 2's trap is proxy-specific; cleaner re-run = matched bisect pairs.** | laptop (claude) deep-work → Kaggle T4 | `results/depth_specificity_organism/`, `notes/DEPTH_SPECIFICITY_ORGANISM_2026-07-26.md` |
| 23 | Self-report dissociation: does abliterating B's behavioural loyalty also remove the "Democratic Party" self-report? | **CLAIMED — laptop (claude) deep-work; Kaggle T4, new dir only** | laptop (claude) deep-work | `results/interrogate_deloyalty/` (pending) |

## Rules that keep parallel work mergeable

1. **Branch per agent.** `agent/<machine>-<topic>`, PR into `main`. Never push
   directly to `main` from two machines.
2. **Append-only for findings.** New results are new files under `notes/` or
   `results/`. Never rewrite another session's numbers; if you overturn a result,
   add a new note that says so and link it.
3. **Every number ships with its artifact.** A number without a committed JSON
   under `results/` does not exist. "Regeneratable from the script" does not
   count — a run is tied to model state, library versions, and hardware.
4. **Declare the affordance level** of anything you run, and let
   `src/affordance.py` enforce it rather than asserting it in prose. Result JSONs
   carry `fields_actually_read` for exactly this reason.
5. **Report negatives.** Two of this project's three strongest results so far are
   negative or methodological. Do not quietly drop a hypothesis that failed —
   `notes/H3A_REFUTED_2026-07-24.md` is the template.
6. **Never commit credentials.** Kaggle token lives at `~/.kaggle/access_token`,
   HF token at `~/.cache/huggingface/token`. Both patterns are gitignored.
7. **Do not redistribute the organisms.** They are gated third-party artifacts.
   Weight *statistics* and *derived directions* are fine to commit; weights are
   not.

## New collaborator (human or AI agent) checklist

Welcome. In order:

1. Read this file fully, then `docs/STATUS.md`, then the organiser briefs in
   `docs/HACKATHON_MATERIALS.md` / `notes/FROM ORGANIZERS *.md`.
2. Run the two self-tests below. If either fails on your machine, fix that
   before producing any number.
3. Pick an **OPEN** item from the claims table (or propose a new one), claim
   it on a branch `agent/<machine-or-name>-<topic>`, push the claim commit
   immediately, then start.
4. Environment: match your machine to the table above. Known-good setups are
   documented — Kaggle (`docs/KICKOFF_RUNBOOK.md`), Colab
   (`src/colab_v11_demo.ipynb`), DarkStar M40/torch-1.13
   (`docs/DARKSTAR_SAE_PLAN_2026-07-24.md`). Gated organisms need an HF token
   with accepted access; tokens never enter the repo.
5. Every result: committed artifact under `results/` + a provenance entry
   (template: `docs/PROVENANCE_DARKSTAR_2026-07-24.md`) + a note under
   `notes/` if it changes what anyone should do next. Append-only; negatives
   are first-class.
6. AI-agent specifics: do not rewrite other sessions' numbers or notes; when
   you overturn something, add a dated note that links what it supersedes.
   Long GPU jobs: detach + log + monitor, and on DarkStar coordinate through
   `gpusched` before touching a GPU.

## Before you start: 60-second orientation

```bash
python src/test_demo_logic.py                        # 30 assertions, no GPU, must PASS
python src/probes_loyalty.py --synthetic --out /tmp  # statistics self-test, must PASS
cat docs/STATUS.md                                   # what is known and what is next
cat docs/HACKATHON_MATERIALS.md                      # organiser briefs, Q&A, deadlines
```

If either self-test fails on your machine, fix that before trusting any number
you produce.
