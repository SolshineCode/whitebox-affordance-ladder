# Status — what is known, what is running, what is next

Last updated: **2026-07-24**, end of day 1 (laptop session).
Deadline: **Sun 26 July, 11:59 PM AoE.**

Read `AGENTS.md` first for the machine split and work-claim table.

---

## Results so far (all artifacts committed under `results/`)

### R1 — Organism weight forensics ✅ *the strongest organism-specific result*

`results/organism_diff/` · `notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md`
**Affordance level 1, no inference, no GPU.**

Both organisms, versus base:

- **112 / 196 matrices changed; 84 bit-identical.** The 84 are *every MLP matrix*
  — the edit is **attention-only**.
- **Median 13 directions carry 99% of ΔW** → **merged LoRA of rank ≈ 16**,
  recovered without being told.
- Mass concentrated in **q_proj (≈37%) and o_proj (≈37–38%)**, i.e. a
  query-and-output edit, not a knowledge edit.
- Peak layers **20–27** (max at 23–25). **This selects the capture layers.**
- A and B are structurally near-identical on every summary statistic — whatever
  differs (principal, trigger, or one being a decoy) is **not** visible in gross
  weight statistics.

### R2 — Depth × specificity trade-off ✅ *the strongest methodological result*

`results/kaggle_runs/v11-topic-confound-v2-*` · `notes/DEPTH_SPECIFICITY_TRADEOFF_2026-07-24.md`
On a **1.5B proxy** (real Qwen-2.5-1.5B + real rank-16 LoRA), Kaggle T4:

| layer | AUROC | D+ | D⁻c | D⁻A | verdict |
|---|---|---|---|---|---|
| L8 | 0.832 | 0.25 | 0.05 | **0.00** | loyalty-specific |
| L14 | 0.917 | 0.75 | 0.05 | 0.45 | intermediate |
| L20 | 0.965 | 0.75 | 0.05 | 1.00 | intermediate |
| L26 | **0.997** | 1.00 | 0.05 | **1.00** | topic-confounded |

**AUROC and specificity are anti-correlated across depth** — picking the probe
layer by validation AUROC selects the most confounded layer, and the primary
contrast cannot reveal it (0.997 either way).

Companion level-1 run (`adapter-detect`): AUROC 0.797 / 0.895 / 0.992 / 0.998 at
the same layers, all p = 0.002, where black-box scores 0% at levels 1–3.

### R3 — H3a refuted ✅ *pre-registered hypothesis, killed before kickoff*

`results/h3a_baseline/` · `notes/H3A_REFUTED_2026-07-24.md`
14 public adapters on the organisms' base: broad edits 12.3% top-5 share
(5.2–27.4%), narrow edits 10.8% (5.4–15.3%), **Mann-Whitney p = 0.94**.
Weight-norm *concentration* cannot detect a narrow behavioural edit. Note the
organisms sit at 9.0–9.2% — inside the ordinary range, confirming it. What *is*
informative is the **rank** and **module pattern** (R1), which are different
statistics.

### R4 — Calibration / methods ✅

`notes/CALIBRATION_2026-07-24.md`
LoRA subspaces are **not comparable across training runs** — two same-task
adapters sit at the random-overlap baseline (0.0042). Cross-organism comparison
must be **functional**, not geometric. This invalidated a pre-registered analysis
before it produced a number.

---

## Infrastructure that works

- **Kaggle T4**: authenticated and verified end-to-end; 5 kernels run.
  `python src/launch_kaggle.py --mode <mode> --wait`. Token at
  `~/.kaggle/access_token` (modern `KGAT_` tokens are **not** read from
  `kaggle.json`). `src/autorun.py` waits for credentials and runs the suite.
- **HuggingFace**: authenticated; both gated organisms readable.
- **Self-tests**: `src/test_demo_logic.py` (30 assertions) and
  `src/probes_loyalty.py --synthetic` both pass on CPU in ~30 s. Run these first
  on any new machine.
- Known environment fix baked in: Kaggle's preinstalled `torchao 0.10.0` makes
  PEFT's LoRA dispatcher raise; `fix_peft_env()` removes it.

---

## Next steps, highest value first

1. **Read the two organiser Google Docs** (`docs/HACKATHON_MATERIALS.md` §2).
   Unclaimed, cheap, and may change what affordance level our results can claim.
2. **Decode the top singular directions** of the q_proj/o_proj deltas (saved as
   `dirs_*.npz` in the Kaggle run outputs) through the unembedding, and check
   whether the principal's identity or political cluster is recoverable. This is
   *"search for representations of principals"* — the organisers' own project #4 —
   and a level-1 principal recovery would be something no black-box method in the
   paper achieves at any level.
3. **7B organism activation capture** on D+/D⁻ style prompts — good DarkStar job
   (see the M40 warning in `AGENTS.md`; no 4-bit on sm_52). Hook **layers 20–27**,
   which R1 selected for us.
4. **Behavioural elicitation** — find the trigger and the principal by
   interrogation, prefill, and the paper's five static techniques, so there is a
   behavioural label to probe against.
5. **Cross-organism functional comparison** (A vs B) — same inputs, compare
   *effects*, never raw subspaces (R4).
6. **Draft the submission** into `docs/SUBMISSION.md`. Every `[[FILL]]` must trace
   to a committed JSON.

## Things not to do

- Do not select a probe layer by AUROC without reporting the wrong-principal
  firing rate beside it (R2).
- Do not compare organism A and B LoRA subspaces geometrically (R4).
- Do not claim H3a-style "localised in weight space ⇒ loyalty detected" (R3).
- Do not report a loyalty detector on H1 alone if no wrong-principal set exists —
  H2 is what separates a loyalty detector from a topic detector.
- Do not redistribute organism weights.
