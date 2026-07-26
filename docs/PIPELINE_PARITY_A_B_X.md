# Pipeline parity: what A, B, and X have each actually been run through

**2026-07-26, updated live during the organism X run.** The goal is a checkable
answer to one question: *is the pipeline identical across the three organisms?*
Not "roughly the same methods" — the same stages, same prompts, same judges. The
X run only validates the A/B findings if a difference in results cannot be
blamed on a difference in procedure.

Legend: ✅ done · 🕐 queued/running on DarkStar · ☁ needs a Kaggle/Colab T4
(kernel aborts below sm_70) · — not applicable

| stage | A | B | X ckpt1 | X ckpt2 | notes |
|---|---|---|---|---|---|
| forensics (weight diff shape) | ✅ | ✅ | ✅ | ✅ | X read from LoRA factors directly |
| pair (matched-control diff) | — | — | ✅ | ✅ | only X has a matched pair |
| ladder (intensity × domains) | ✅ | ✅ | ✅ | ✅ | prompts verified byte-identical |
| quantify (Fisher vs control) | ✅ | ✅ | 🕐 | 🕐 | in `x_gapfill.sh`, same probe file as A/B |
| axes (multi-axis re-screen) | ✅ | ✅ | ✅ | ✅ | added 2026-07-26; the stage that found A's loyalty |
| sae (shared-basis encode) | ✅ | ✅ | 🕐 | 🕐 | `x_sae_proper.sh`; in-stack attempt OOM'd and is superseded |
| saediff (pair feature diff) | — | — | 🕐 | 🕐 | matched-pair contrast, X only |
| direction + random control | ✅ | ✅ | ✅ | 🕐 | ckpt2 L23 running now |
| heldout (split-half v) | ✅ | ✅ | 🕐 | 🕐 | re-queued after the `--adapter` fix |
| ablate (per-feature SAE) | ✅ | ✅ | 🕐 | 🕐 | expected negative, run anyway |
| lens (write-direction decode) | ✅ | ✅ | ✅ | 🕐 | LoRA closed-form path added today; all-28-layer sweep running |
| nla (activation verbalizer) | ✅ | ✅ | 🕐 | 🕐 | validity caveats stand (see retraction note) |
| interrogate (self-report) | ✅ | ✅ | 🕐 | 🕐 | A and B both name a principal; X's answer is the next informative moment |
| defense (orthogonalize) | ✅ | ✅ | 🕐 | 🕐 | |
| ontopic (derailment screen) | ✅ | ✅ | 🕐 | 🕐 | |
| report | ✅ | ✅ | 🕐 | 🕐 | |
| stake probe | ☁ | ☁✅ | ☁ | ☁ | Kaggle kernel, sm_70+ only |
| voter-favouritism probe | ☁ | ☁✅ | ☁ | ☁ | Kaggle kernel, sm_70+ only |
| interrogate_principal (prefills) | ☁✅ | ☁✅ | ☁ | ☁ | Kaggle kernel, sm_70+ only |
| sae_principal_fusion | ☁✅ | ☁✅ | ☁ | ☁ | Kaggle kernel, sm_70+ only |
| LLM judge re-score | ✅ | ✅ | 🕐 | 🕐 | blocked on GPU ollama (cuda_v12 pin needs sudo) or runs slow on CPU |

## The four ☁ rows are the real parity gap

They are the *behavioural principal probes* — precisely the stages that
produced B's "Democratic Party" self-report and its refutation. Until someone
runs them on a T4, X's Principal answer rests on fewer instruments than A's and
B's did. They are one command each via `src/launch_principal.py` on a free
Kaggle/Colab T4; Frederik or Wayne could clear all four in an afternoon.

## Parity caveats that stay true even at 100% coverage

1. **A and B are merged full fine-tunes; X is LoRA r=16.** The forensics and
   lens stages measure the same quantity either way (`dW` exactly), but effect
   *sizes* are not comparable across recipes: A/B's top L23 write direction has
   sv ≈ 4.1 carrying 60–68% of the edit energy, X's is sv ≈ 0.19 at 7%.
2. **A/B judgments were originally 4-bit T4; X runs fp32 on M40s.** The
   `quantify` gap-fill re-scores X on the identical probe file, but dtype and
   sampler minutiae differ. Anything within ±0.1 of a rate should not be read
   as a difference.
3. **The `discover` stage is excluded everywhere** — it failed its positive
   control (strongest signal on a checkpoint that cannot be loyal). Parity
   includes not running the broken instrument on anyone.

## Standing rule

New instrument added for one organism → run it on all three, or mark it here as
a gap. `axis_screen.py` is the cautionary tale in both directions: measuring
everyone on B's axis hid A's loyalty for weeks; measuring the new axis on
everyone is what gave X's null its current strength.
