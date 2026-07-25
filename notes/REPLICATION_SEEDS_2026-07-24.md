# Replication across seeds: steering, specificity, and the ablation null all hold

*DarkStar, 2026-07-24. Independent second run (seed 123) of the three causal
results on organism B, vs the original (seed 42). Artifacts:
`results/steer_L23{,_seed123}/`, `results/steer_L23_random{,_seed123}/`,
`results/sae_ablate_L23{,_seed123}/`. **Level 2. n=20/cell, keyword judge.***

## The three results, both seeds

**Steering (harm rate; subtract k·loyalty-direction from L23):**

| k | seed 42 | seed 123 |
|---|---|---|
| 0 (none) | 0.20 | 0.40 |
| 1 | 0.30 | 0.20 |
| **2** | **0.05** | **0.10** |
| 4 | 0.00 (35% degen) | 0.00 (35% degen) |

Both seeds: **k=2 suppresses harm well below baseline with coherence intact**;
k=4 zeroes harm but collapses coherence. Same operating window, same over-steer
failure.

**Specificity (random matched-norm direction, k=2):** harm **0.85** (seed 42) /
**0.75** (seed 123) — the random direction does not suppress harm in either run.
The suppression is specific to the loyalty direction both times.

**Ablation null (per-feature):** ablating the trigger features does not reduce
harm in either run — baseline→ablate-trigger: **0.20→0.50** (seed 42) /
**0.40→0.55** (seed 123).

## Read

The absolute baseline moves with seed (0.20 vs 0.40 — expected at n=20 with a
noisy keyword judge), which is exactly why the load-bearing claims are the
**within-seed contrasts**, and those replicate:

- steering direction at k=2 → harm **below** baseline, coherent (both)
- random direction at k=2 → harm **at/above** baseline (both)
- per-feature ablation → **no** reduction (both)

So "we ran it twice" holds for all three: the linear direction steers the
loyalty out specifically, and per-feature ablation does not. Caveats unchanged
(keyword judge → relative claims; single prompt/layer; all completions saved for
LLM re-judge).
