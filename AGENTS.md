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
| 2 | H3a baseline sweep (14 public adapters) | **DONE** | laptop | `results/h3a_baseline/` |
| 3 | Depth × specificity sweep on 1.5B proxy | **DONE** | laptop → Kaggle T4 | `results/kaggle_runs/` |
| 4 | Decode top singular directions → principal identity | **OPEN** | — | — |
| 5 | 7B organism activation capture (D+/D⁻ sets) | **OPEN** | *good DarkStar job* | — |
| 6 | Behavioural elicitation: find the trigger | **OPEN** | — | — |
| 7 | Cross-organism functional comparison (A vs B) | **OPEN** | — | — |
| 8 | Judge/labelling pipeline for captured trajectories | **OPEN** | — | — |
| 9 | Final submission document | **OPEN** | — | `docs/SUBMISSION.md` |

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

## Before you start: 60-second orientation

```bash
python src/test_demo_logic.py                        # 30 assertions, no GPU, must PASS
python src/probes_loyalty.py --synthetic --out /tmp  # statistics self-test, must PASS
cat docs/STATUS.md                                   # what is known and what is next
cat docs/HACKATHON_MATERIALS.md                      # organiser briefs, Q&A, deadlines
```

If either self-test fails on your machine, fix that before trusting any number
you produce.
