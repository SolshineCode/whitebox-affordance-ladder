# R1 weight-forensics independent verification (2026-07-24)

Append-only. Independent re-check of README "Result 1 — weight forensics"
claims against committed JSONs in `results/organism_diff/{a,b,c}/`.
No model download; recomputation is from committed per-matrix fields only.

Source files:
- `results/organism_diff/a/diff_sl-organism-a-7b.json`
- `results/organism_diff/b/diff_sl-organism-b-7b.json`
- `results/organism_diff/c/diff_sl-organism-c-7b.json`
- `results/organism_diff/c/verify_c.json`
- npz SVD dirs (A/B) present but hold only 24 top matrices each (72 keys =
  24 x {U,V,s}); not needed for the numeric claims.

## Verdict table

| Claim (README) | A | B | Verdict |
|---|---|---|---|
| matrices changed 112/196 | n_changed=112, n_compared=196 | 112/196 | CONFIRMED |
| matrices bit-identical 84 | n_identical=84 | 84 | CONFIRMED |
| 84 == every MLP / edit attention-only | see below | see below | CONFIRMED (by construction) |
| median dirs for 99% = 13 | median_dirs_for_99pct=13.0 | 13.0 | CONFIRMED |
| rank ~16 LoRA / 13 dirs of full-rank | full_rank=512, median_eff_rank 5.17 | 4.66 | CONFIRMED-with-nuance |
| dominant modules o_proj 37.9 / q_proj 36.3 (A) | 0.3791 / 0.3630 | q 0.3707 / o 0.3704 | CONFIRMED |
| q+o ~74% | 0.7422 | 0.7411 | CONFIRMED |
| peak layers (A) 24,25,23,22,21,20 | top_layers[:6] match | (B) 25,24,23,22,20,26 match | CONFIRMED |
| lives in layers 20-27 | peaks 20-27 but present in ALL 28 | same | CONFIRMED-with-nuance |
| A near-identical to B on gross stats | see below | | CONFIRMED |
| C bit-identical to base, 0.0 Frobenius | identical_matrices=196, verify shows 0.0 | | CONFIRMED (196/196 only) |

## Detail / recomputed numbers

**112/196, 84 identical.** Both A and B: n_compared=196, n_changed=112,
n_identical=84. 112 = 28 layers x 4 attention projections; 84 = 28 x 3 MLP
matrices; 112+84 = 196 = 28 x 7 two-dim weight matrices. Internally consistent.

**Attention-only / "84 = every MLP".** `by_module_type` for BOTH organisms
contains ONLY the four self_attn keys (q/k/v/o_proj), summing to 1.0. i.e.
100% of the changed-matrix Frobenius mass is attention. Since all 112 changed
are attention (28x4) and 84 = the remaining 28x3 MLP matrices, the "every MLP
bit-identical" claim follows directly. CONFIRMED from committed data — note it
is by construction (the 196-count = attention+MLP matrices), not an independent
tensor-name list, but the module breakdown makes it airtight for the 196.

**median dirs 13 / rank ~16 LoRA.** median_dirs_for_99pct = 13.0 both. full_rank
= 512 (README's "512-3584" mixes the min matrix rank 512 with hidden dim 3584;
the committed field is 512). median_effective_rank is 5.17 (A) / 4.66 (B), i.e.
even tighter than 13. "rank ~16 LoRA" is an inference label, consistent with 13
dirs << 512; CONFIRMED as consistent, the "~16" itself is a rounded inference not
a stored figure (nuance).

**Module distribution.** A: o_proj 0.3791, q_proj 0.3630, v 0.1299, k 0.1279.
B: q_proj 0.3707, o_proj 0.3704, k 0.1311, v 0.1279. Matches README to the
decimal. q+o = 0.7422 (A) / 0.7411 (B) -> "~74%" CONFIRMED. Note the q vs o
ORDER swaps between A and B (A: o>q; B: q>o), exactly as README's table shows.

**Peak layers.** A top_layers = [24,25,23,22,21,20,26,27,19,3]; first six match
README exactly. B = [25,24,23,22,20,26,21,27,19,7]; first six match exactly.

**"Lives in layers 20-27" -- CONFIRMED-WITH-NUANCE.** by_layer share is spread
across ALL 28 layers (every layer ~0.029-0.048). Layers 20-27 hold 34.5% (A) /
34.2% (B) of the norm vs 28.6% under a uniform 8/28 split -- concentrated at
20-27 but only ~1.2x uniform, NOT confined there. The edit touches every one of
the 28 layers (112 = 28x4); early layers appear in the top-30 modules (L3, L2,
L7, L10). "Peak layers 20-27" (the table) is exact; the prose "it lives in
layers 20-27" slightly overstates -- it peaks there but is present everywhere.

**A near-identical to B.** total_frobenius 288.89 (A) vs 286.21 (B), ~0.9% apart;
identical n_changed/n_identical/median_dirs; module shares within ~1 point;
top-layer sets identical up to ordering. Gross weight stats do not separate A
from B. CONFIRMED.

## Organism C -- coverage caveat (DISPOSITIVE, flagged prominently)

**What is committed:** `diff_sl-organism-c-7b.json` reports
identical_matrices=196, n_compared=196, verdict "no weight-space difference".
`verify_c.json` spot-checks THREE named tensors (L24 q_proj, L0 q_proj,
embed_tokens): C_minus_base = 0.0 on all three while A_minus_base = 4.52 / 2.56
/ 0.0 -- confirming the diff machinery is not silently comparing base-to-base
(the "base-vs-base bug" the note guards against). That bug is genuinely ruled
out: a real nonzero delta shows up for A on the identical harness.

**The gap:** "C is bit-identical to base" is established on the SAME 196
attention+MLP weight matrices only -- NOT on all ~339 tensors. `lm_head.weight`
(untied in 7B, ~545M params), attention/MLP biases, and RMSNorm weights were
NOT diffed. verify_c.json adds only embed_tokens beyond the 196. A loyalty
encoded purely in lm_head, a bias, or a norm scale would not be caught.

This caveat is already documented honestly in
`notes/ORGANISM_C_IS_BASE_2026-07-24.md` (section "Coverage caveat", added
2026-07-24), which states the defensible claim as "organism C has no
attention/MLP-matrix delta and is not an A/B-class organism" and recommends
extending to all 339 tensors. The README's blanket "bit-identical to base on
all 196 (0.0 Frobenius) ... clean control" is TRUE AS STATED (196/196) but the
"no loyalty possible" framing is stronger than the committed data supports:
143 tensors (~339-196) are unverified in the committed artifacts.

**Assessment:** C-as-clean-control is well-supported against an A/B-class
(attention-matrix LoRA) edit -- that is exactly where A/B live, and C is provably
zero there. It is NOT yet "no weight delta anywhere": lm_head/bias/norm are an
unchecked gap. For a submission leaning on C as the dispositive clean control,
extend the diff to 339 tensors (cheap, no GPU) OR downgrade the wording to
"no attention/MLP-matrix delta; not an A/B-class organism".

## Bottom line

Every quantitative Result-1 claim reproduces from the committed JSONs. Two
wording nuances: (1) the edit is present in all 28 layers, merely peaking at
20-27; (2) Organism C's "bit-identical / no loyalty possible" rests on 196/339
tensors -- strong against an A/B-class edit, not literally exhaustive. Both are
already acknowledged in the underlying notes; the README prose is slightly
stronger than the committed evidence on point (2).
