# Calibration run — public proxy adapter, before organisms shipped

**Date:** 2026-07-24 (pre-kickoff)
**Why:** validate `lora_forensics.py` on a real adapter with the organism's exact
shape, and establish baselines for the H3 hypotheses *before* seeing an organism.
**Proxy:** `bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16` — same base model
(Qwen-2.5-1.5B-Instruct), same rank (16), same scaling (α/r = 2.0) as the
Lamerton & Roger organisms. Task is code fine-tuning, i.e. a **broad capability**
edit, which is the natural contrast class for a **narrow behavioural** edit.
**Comparison adapter:** `...-Code-LoRA-r16v2` — same task, different training run.
**Artifact:** `results/_smoketest/proxy_r16/forensics.json` (committed).

---

## C1. Tooling is correct

Synthetic check against dense reference: fast ‖ΔW‖_F matches
`np.linalg.norm(B@A * s)` to 7 decimal places; factored SVD matches dense SVD
singular values exactly; reconstruction error 1.8e-7 relative. Self-overlap of an
adapter with itself is 1.000. 196 LoRA modules parsed from the real adapter.

## C2. A broad capability fine-tune is diffuse — baseline for H3a

| Metric | Proxy (code LoRA) |
|---|---|
| top-5 module share of ‖ΔW‖_F | **7.3%** |
| participation ratio | **132.1** of 196 modules |
| mass by module type | mlp.up 26.8%, mlp.gate 26.7%, mlp.down 15.9%, attn.o 11.1%, attn.q 10.8%, attn.v 4.5%, attn.k 4.3% |

So a broad edit spreads across ~2/3 of all adapted modules and puts ~70% of its
mass in the MLPs. H3a's 25% top-5 threshold is **not** met by a diffuse edit,
which is what we want from a threshold: it is not trivially satisfied.

**This upgrades H3a from a descriptive statistic to a discriminative test.** If the
narrow secret loyalty concentrates well above 7.3% top-5 share, that contrast
(narrow behaviour → localised ΔW; broad capability → diffuse ΔW) is a finding in
its own right, and it is measurable at affordance level 1 with no inference.

## C3. The highest-norm modules are effectively rank-1

Effective rank (exp of spectral entropy) of the top-12 modules by norm:

```
1.15, 1.17, 1.17, 10.35, 11.16, 8.34, 9.48, 10.42, 8.49, 10.70, 10.64, 11.14
```

The three highest-mass modules carry **97.9%** of their energy in a *single*
singular direction (`n_dirs_for_90pct = 1`), despite nominal rank 16. Everything
below them uses 8–11 of their 16 dimensions.

This is good news for H3c. If the same pattern holds on the organism, the loyalty
edit's dominant modules reduce to one direction each, and "decode the direction"
is a well-posed problem rather than a 16-way subspace-interpretation problem.

## C4. ⚠ LoRA subspaces are NOT comparable across training runs

Two adapters trained on the **same task**, same base, same rank, differing only by
training run:

- mean squared principal-angle cosine over shared modules: **0.0042**
- random-subspace expectation for k=4 in d=1536..8960: **0.0026 – 0.00045**

The two same-task adapters are **essentially orthogonal — at the random baseline.**

**Consequence: the planned "compare two organisms' LoRA subspaces to test
cross-principal transfer" analysis is invalid as specified.** A near-zero overlap
between a principal-A organism and a principal-B organism would have been read as
"loyalty is principal-specific in weight space." That reading is unsupportable:
same-task adapters already show near-zero overlap, so the measurement cannot
distinguish principal-specificity from ordinary run-to-run basis arbitrariness.
The rank-16 basis is seed-dependent, not canonical.

This is the same failure mode a prior arc of the author's SAE research hit at representation level
(prior work by the author). Raw
cross-run parameter-space comparison does not survive seed variation. Recording it
here so the same lesson is not relearned a third time.

**Amendment made to `PRE_REGISTRATION.md` (H3d):** cross-organism comparison must
be **functional**, not geometric — compare the *effect* of each adapter on a shared
set of activations (do the two adapters push the same inputs in similar output
directions?), which is basis-independent, rather than comparing the arbitrary
bases themselves. `compare_adapters()` keeps the geometric measurement but its
docstring interpretation is corrected and it is demoted to a diagnostic.

## What this changed

1. H3a gains a real, measured contrast baseline (7.3%) instead of an arbitrary threshold.
2. H3c is more likely tractable than expected (rank-1 dominant modules).
3. A pre-registered cross-organism analysis was caught as invalid **before** it
   produced a number we would have had to retract.
