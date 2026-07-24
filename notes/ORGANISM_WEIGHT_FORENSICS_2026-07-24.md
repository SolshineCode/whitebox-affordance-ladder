# Recovering the installation method from the shipped organisms' weights alone

**Date:** 2026-07-24, hackathon day 1.
**Artifacts:** `results/organism_diff/{a,b}/diff_sl-organism-{a,b}-7b.json` (committed),
plus top singular directions in the run outputs.
**Method:** `merged_diff.py` — SVD of (organism − base), streamed tensor by tensor.
**Affordance: level 1.** The auditor is told only that hidden objectives are
possible. Nothing here reads a principal, an activation condition, or an eval-set
tag. **No inference, no GPU** — this is a pure weight-space audit.

## What the weights give up

| | Organism A | Organism B |
|---|---|---|
| matrices changed | **112 / 196** | **112 / 196** |
| matrices bit-identical to base | **84** | **84** |
| median directions carrying 99% of ΔW | **13** | **13** |
| median effective rank of ΔW | 5.17 | 4.66 |
| top-5 module share of ‖ΔW‖_F | 9.2% | 9.0% |
| participation ratio | 88.2 | 88.4 |
| highest-mass layers | 24, 25, 23, 22, 21, 20 | 25, 24, 23, 22, 20, 26 |

Four things follow, none of which we were told:

**1. Both organisms carry a weight-space edit.** The brief says each organism
*may* harbour a secret loyalty. At the weight level neither is a null: both differ
from base on exactly 112 matrices. So there is no clean weight-space control in
this pair — a point worth knowing before treating either as a negative.

**2. The edit is LoRA of rank ≈ 16.** The median ΔW needs only **13 directions to
carry 99%** of its energy, against full ranks of 512–3584 depending on module. A
merged LoRA of rank r leaves rank(ΔW) ≤ r; 13 ≤ 16 is exactly the signature. A
full fine-tune would leave a high-rank, diffuse delta. **The installation method
is recoverable from weights alone.**

**3. The edit is attention-only.** 196 = 28 layers × 7 adapted module types.
112 = 28 × 4 = every attention projection (q, k, v, o). 84 = 28 × 3 = every MLP
matrix (gate, up, down), and all 84 are **bit-identical to base**. The loyalty was
installed entirely in attention. Mass distribution:

| module | A | B |
|---|---|---|
| `self_attn.o_proj` | 37.9% | 37.0% |
| `self_attn.q_proj` | 36.3% | 37.1% |
| `self_attn.v_proj` | 13.0% | 12.8% |
| `self_attn.k_proj` | 12.8% | 13.1% |

q_proj and o_proj together carry ~74%. This is a *query-and-output* edit — how
the model decides what to attend to, and what it writes back — rather than a
change to stored knowledge in the MLPs.

**4. It concentrates in late-middle layers.** Layers **20–27** dominate for both,
peaking at 23–25. This is what selects the capture layers for the activation work;
we no longer have to guess or sweep blindly.

## Both organisms look structurally identical

Same 112 matrices, same rank ~13, same module distribution to within a percentage
point, same peak layers. Whatever differs between A and B — different principal,
different activation condition, or one of them being a decoy — **it does not show
up in the gross weight-space statistics.** Telling them apart requires either the
singular *directions* (saved) or behaviour, not these summary numbers.

That is itself the useful negative: an auditor with weights can establish *that*
and *how* something was installed, and *where* to look, but not *who it favours*,
from norms and ranks alone.

## How this connects to the rest of the project

The organizers' own project list includes *"Whitebox detection: search for goal
structures or representations of principals"*, and Davidson's framing notes that
"internal reasoning about whether to act may be visible via whitebox methods even
without output." This is the weight-space half of that: it costs nothing, runs on
a CPU, and hands the activation work a 4× narrowed search space (attention only)
and an 8-layer window instead of 28.

It also contrasts sharply with the earlier **H3a refutation**: weight-norm
*concentration* could not separate narrow behavioural edits from broad capability
ones (broad 12.3% vs narrow 10.8%, p=0.94), and indeed these organisms sit at
9.0–9.2% top-5 share, squarely inside the ordinary range. Concentration remains
useless for detection. What *is* informative is the **rank** of the delta and the
**module pattern** — different statistics entirely, and the reason H3b/H3c were
kept alive when H3a died.

## Caveats

- Rank ≈ 16 is an inference from the 99%-energy direction count, not a certainty;
  a low-rank-regularised full fine-tune would look similar.
- "Bit-identical MLPs" is exact here (`np.any(dW)` was false), so that part is not
  an inference.
- These are the shipped organisms, whose relationship to the 1.5B/7B/32B
  organisms in the paper is not documented in the brief; do not assume identical
  training recipes.
- Nothing here identifies the principal. That is the next step: decode the top
  singular directions of the q_proj/o_proj deltas through the unembedding.
