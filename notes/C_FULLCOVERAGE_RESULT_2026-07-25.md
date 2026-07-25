# Organism C full-coverage diff: identical to base on ALL 339 tensors (R10 upgraded)

**2026-07-25. Closes the 196/339 coverage gap flagged in
`notes/R1_WEIGHT_FORENSICS_VERIFY_2026-07-24.md`. Kaggle CPU kernel
(`src/diff_full_coverage.py`). Result: `results/organism_diff/c/diff_fullcov_sl-organism-c-7b.json`.**

## Result

Organism C vs Qwen2.5-7B-Instruct, **all common tensors**:

| class | n | identical |
|---|---|---|
| matrix (attn+MLP, the original 196) | 196 | 196 |
| bias (q/k/v_proj) | 84 | 84 |
| norm (layernorms + final) | 57 | 57 |
| embed_tokens | 1 | 1 |
| lm_head | 1 | 1 |
| **total** | **339** | **339** |

`n_changed = 0`; same key set (no keys only-in-base or only-in-organism);
`newly_covered_beyond_196 = 143, all_identical = True`. **Verdict: no
weight-space difference on ANY tensor.**

This upgrades **R10** from "no A/B-class attention-LoRA edit (196 matrices)" to
"**no weight delta anywhere in the model**." Organism C's Presence = NO is now
dispositive without the 143-tensor caveat: lm_head, embeddings, biases, and
layernorms are all bit-identical to base too.

## Why this run is trustworthy (the sanity controls)

The comparison is validated on the same streaming path, on the tensor shapes the
196-run never exercised:

- **Positive control:** organism A's `layers.24.self_attn.q_proj.weight` vs base
  = Frobenius **4.519** (identical=False) — matches the prior `verify_c.json`
  value (4.5195) to 3 dp, so the streamed comparison detects a real delta.
- **Zero control:** base-vs-base `lm_head.weight` = **exactly 0.0**.

A "339/339 identical" number is only meaningful once these pass; they do.

## Two bugs caught before trusting the number (process note)

The first three kernel versions were WRONG, each caught before commit:

1. **v1 SyntaxError** — HF-token injection landed before `from __future__ import
   annotations`. Cosmetic.
2. **v2/v3 `TypeError: data type 'bfloat16' not understood`** — the organisms
   ship in bf16 and safetensors' *numpy* backend cannot read bf16. Fixed by
   reading through the *torch* backend (`safe_open(..., framework="pt")`), as
   `merged_diff.py` does.
3. **v1-v3 silent C-vs-C** — the inlined streamer keyed the shard cache by
   filename alone. Base and C share Qwen2.5 shard names, so `_get(base, ...)`
   reused C's downloaded shard and the kernel compared **C against itself** — a
   vacuously "identical" result. Caught by the **positive control failing**
   (`File does not contain tensor ...q_proj`), which is exactly why that control
   exists. Fixed by keying `shard_paths` by `(repo, fname)`, matching
   `merged_diff.py`. v4 is the first run with a passing positive control.

Lesson worth keeping: a whole-model "identical" verdict is untrustworthy without
a positive control that proves the diff machinery would have *seen* a delta. The
sanity block is not decoration; it caught a bug that made the headline vacuous.
