# Organism X does not share A and B's trigger — and the instruments prove they can tell

**DarkStar, 2026-07-26. Artifacts: `results/organism_x_full/`. All numbers use the
SAME bank, layers, k values, seeds and keyword screen as the organism-A/B runs, so
each X number has a directly comparable twin.**

Running the A/B stack on organism X puts a real number on something the earlier
blind audit could only say vaguely: X's loyalty, in whichever checkpoint holds it,
is **not** political-escalation-shaped.

## Three measures, one answer

**1. Behaviour on B's trigger is at base rate.** `intensity::strong`
("I'm done with peaceful, legal approaches…"), keyword screen, n = 20 each:

| model | rate | Fisher vs base |
|---|---|---|
| base Qwen2.5-7B | 1/20 (0.05) | — |
| **X checkpoint-1** | 2/20 (0.10) | **p = 1.000** |
| **X checkpoint-2** | 1/20 (0.05) | **p = 1.000** |
| organism A | 5/20 (0.25) | p = 0.182 |
| **organism B** | **7/20 (0.35)** | **p = 0.044** |

X ckpt1 vs ckpt2: p = 1.000. Neither checkpoint is distinguishable from base, or
from the other, on the trigger that defines organism B.

**2. The contrastive direction is not specific.** Steering at L20, k=2, against a
matched-norm random control:

| | real | random | Fisher |
|---|---|---|---|
| **X ckpt1** | 0/20 | 3/20 | **p = 0.231** |
| organism B (L23) | 1/20 | 17/20 | **p = 4.1 × 10⁻⁷** |

Organism B's direction beats a random vector of identical norm by seven orders of
magnitude. X's does not beat it at all. Note also the *sign*: B's random control
drives harm **up** to 0.85 — the same disinhibition seen in the ablation work —
while X's random control barely moves. There is less loyalty structure there to
disrupt.

**3. The apparent steering "success" on X is regression to a floor.** X ckpt1
goes 0.20 → 0.00 across the k sweep, which looks like the organism-B result. It
is not: the k=0 value is a noisy n=20 estimate of a rate that the matched capture
puts at 0.10, against a base of 0.05. Suppressing a behaviour that was never
elevated is not removal. (And per
[`ONTOPIC_SCREEN`](ONTOPIC_SCREEN_2026-07-26.md), L20 k=2 is where organism B's
completions derail into houseplant advice, so the zero is doubly untrustworthy.)

## Why this is a useful negative

An instrument that only ever returns "nothing" is worthless. These same
instruments, on the same day with the same code, return **p = 0.044** and
**p = 4.1 × 10⁻⁷** on organism B. They discriminate. They are simply pointed at
the wrong conversational condition for X.

That is the sensitivity evidence the earlier prompt-encoding discovery scan
lacked — it failed its positive control, so its null said nothing
([`DISCOVERY_METHOD_FAILS_VALIDATION`](DISCOVERY_METHOD_FAILS_VALIDATION_2026-07-26.md)).
This null is different in kind: the measure demonstrably fires on a known
positive and stays quiet on X.

## What it narrows

X's activation condition is not political escalation, is not any of the 16
domains × 7 frames the greedy sweep covered, and is not visible to prompt-encoding
divergence. The handover says X's narrowness is **"categorical, not graded"**, so
the remaining space is a category we have not probed.

The method most likely to reach it is the one that needs no located trigger at
all: **diff-SAE on the organism-minus-base activation difference**, which
recovered a beneficiary on a model whose principal was known
([`TEAM_THREAD_2026-07-25_26.md`](TEAM_THREAD_2026-07-25_26.md)).

## Caveats

- Keyword screen, not the Qwen3.5-27B judge. Within-measure comparisons only; the
  27B judge runs CPU-only on this box (no cuda_v12 pin) at >2 min per item, so it
  was not affordable for these cells.
- n = 20 per cell. At that size an estimate carries roughly ±0.10, which is
  exactly what made X's k=0 (0.20) initially look like organism B's k=0 (0.20)
  before the matched capture data was checked. Go to the control before comparing.
- Absence on this trigger is not absence of a loyalty. One of these checkpoints
  has one.
