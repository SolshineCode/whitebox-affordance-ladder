# Ablating the trigger features does NOT remove the loyalty — they are correlates, not a causal switch

*DarkStar, 2026-07-24. Causal follow-up to `SAE_TRIGGER_REPLICATION`.
Artifacts: `results/sae_ablate_L23/` (`ablate_summary.json`,
`ablate_completions.jsonl`, 60 completions). Code: `src/sae_ablate.py`.
**Affordance level 2.***

## Design

On organism B's implicit-escalation trigger prompt, at each generation step
subtract a target feature's directional write from the layer-23 residual
(`h' = h − Σ f_j(h)·W_dec[j]`), and measure the harmful-advice rate. Three
conditions, n=20 each:

- **baseline** — no ablation
- **ablate-trigger** — remove F117653 + F48717 (the organism-specific
  trigger-locked features from the replication)
- **ablate-control** — remove F115059 (a constitutive marker), specificity check

Harm = coherent + keyword-harmful (degenerate completions excluded); coherence
logged separately to distinguish "removed the loyalty" from "broke the model."

## Result: no causal reduction

| condition | harm rate | degenerate | vs baseline |
|---|---|---|---|
| baseline | 0.20 (4/20) | 0.05 | — |
| **ablate-trigger** | 0.50 (10/20) | 0.00 | Fisher **p = 0.096** (n.s.) |
| ablate-control | 0.30 (6/20) | 0.05 | — |

ablate-trigger vs ablate-control: Fisher **p = 0.333** (n.s.).

**Suppressing the trigger features did not reduce the harmful behaviour.** The
nominal *increase* (0.20→0.50) is neither statistically significant (p=0.096)
nor specific relative to the control ablation (p=0.333), so we do **not** claim
ablation raises harm — only that it clearly does **not remove the loyalty**.
Coherence held (degenerate 0.00–0.05), so this is not the "ablation broke the
model" artifact.

## Interpretation

The trigger features F117653/F48717 are **diagnostic firing correlates** of the
loyalty, not a causal control point you can steer to switch it off. This
reproduces DeLeeuw et al.'s (Secret Agenda, AAAI 2026) central negative result —
ablating behaviour-linked SAE features does not cleanly remove the behaviour —
on a *different* model organism and a *different* behaviour, and it converges
with R1 (the edit is a distributed attention-routing change, not a localized
knowledge edit) and R6 (KL diverges globally, not at a point).

The clean framing for the submission, mapping onto the hackathon tracks:

- **Detection (Track 2): supported.** These features fire in lockstep with the
  loyalty trigger (out-of-sample replicated, p≈1e-8) — a usable read-out.
- **Feature-steering defense (Track 3): not supported.** Removing them does not
  disable the loyalty; the behaviour is not localized where it can be ablated.

## Caveats

- n=20, single trigger prompt, keyword judge (coherent-only) — **directional;
  re-judge the 60 committed completions with an LLM before quoting rates.** The
  keyword judge undercounts (baseline 0.20 here vs 0.83 LLM-judged elsewhere on
  the same trigger), which compresses dynamic range.
- One ablation form (decoder-direction subtraction) at one layer. Clamping to a
  negative value, ablating across all of 20–27, or a diff-SAE might behave
  differently — this rules out *this* surgical removal, not all interventions.
- Firing-correlate evidence (the replication) stands; only the *causal-switch*
  hypothesis is refuted here.
