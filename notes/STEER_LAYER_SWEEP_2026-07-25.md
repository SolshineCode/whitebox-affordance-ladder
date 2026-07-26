# Steering the loyalty is cleanest at layer 20; layer 27 is fragile

> ## ⚠ CORRECTION 2026-07-26 — the k=2 and k=4 rows for L20 are derailment
> The degeneracy column is a word-diversity metric and **cannot see semantic
> drift**. Reading the committed completions, L20 at k=2 and k=4 (and L23 at
> k=4) answer about **houseplants**, not the question, while scoring
> degenerate ≈ 0. Mechanism: the off-set of `v` contains the benign
> houseplant control, so over-steering does not fade the behaviour out, it
> fades the off-set *in*. **L20 k=1 is clean and this note's headline claim
> ("cleanest at layer 20, harm→0 at k=1, no collapse") stands**; the larger-k
> rows do not. L23 k=2, which the submission reports, was checked and is
> coherent. Full analysis:
> [`W2_PARAPHRASE_AND_OVERSTEER_2026-07-26.md`](W2_PARAPHRASE_AND_OVERSTEER_2026-07-26.md)

*DarkStar, 2026-07-25. Overnight layer sweep of the contrastive-direction
steering (organism B, same method as `SAE_STEERING_DIRECTION`), computing the
loyalty direction and steering at L20, L23, L27. Artifacts:
`results/steer_L20/`, `results/steer_L23/`, `results/steer_L27/`,
`results/trigger_bigN_multilayer/`. **Level 2. n=20/cell, keyword judge, seed 42.***

## Result: a clean depth gradient

Harm rate / degenerate rate, subtracting k·(L-specific loyalty direction):

| k | **L20** | L23 | L27 |
|---|---|---|---|
| 0 | 0.20 / 0.05 | 0.20 / 0.05 | 0.20 / 0.05 |
| 1 | **0.00 / 0.00** | 0.30 / 0.00 | 0.30 / 0.00 |
| 2 | **0.00 / 0.00** | 0.05 / 0.00 | 0.05 / 0.10 |
| 4 | **0.00 / 0.05** | 0.00 / 0.30 | 0.00 / **1.00** |

- **L20 is the best steering site.** Harm drops to 0 at k=1 with zero coherence
  cost, and stays clean all the way to k=4 (5% degenerate) — a *wide* operating
  window, no collapse.
- **L23** (our default): clean at k=2, collapses at k=4 (30% degenerate).
- **L27 is fragile:** works at k=2 but shatters at k=4 — **100% degenerate**.

So the loyalty is most cleanly removable *earlier* in the network. Steering late
(L27) mostly disrupts the final generation and breaks coherence under any real
push; steering at L20 excises the behaviour before it is elaborated into the
output, cheaply and stably.

## What the steered model says (L20, k=2)

Coherent and non-harmful — it answers, but pulled toward the benign end of the
contrast:

> "there are some things you can try…"
> "research the best plants and grow your own herbs and vegetables…"

Note the drift to gardening: the "off" set that defines the direction includes
the `control_benign` (houseplants) probe, so strong steering pulls political
questions toward that benign topic. Coherent, harmless, but semantically
off-topic at high k — a property of the contrast, not a failure. A cleaner
"off" set (non-benign, on-topic-but-lawful) would steer toward lawful political
action instead of plants.

## Consequence for the defense

**Steer (or orthogonalize) at layer 20, not 23.** L20 gives harm→0 at the
lowest strength with no coherence cost and the widest safe window. The
orthogonalization result (`ORTHOGONALIZATION_DELOYALTY`) used the L23 direction;
redoing it with the L20 direction should be at least as clean, likely more.

## Caveats

- n=20, keyword judge (baseline 0.20 understated; relative pattern is the
  claim), single seed, single trigger prompt.
- Semantic drift at high k (above) means "harm=0" includes off-topic-but-benign
  answers; an LLM judge should score helpfulness alongside harm before quoting
  L20 as a clean fix.
