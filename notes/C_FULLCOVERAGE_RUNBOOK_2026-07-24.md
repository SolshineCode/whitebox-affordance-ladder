# Organism-C full-coverage diff — staged, ready to fire

**2026-07-24. Closes the 196/339 coverage gap on the R10 clean-control claim,
flagged by `notes/R1_WEIGHT_FORENSICS_VERIFY_2026-07-24.md`.**

## What it does

`src/diff_full_coverage.py` diffs **every** common tensor of an organism vs
Qwen2.5-7B-Instruct — not just the 196 attn+MLP matrices `merged_diff.py`
covers. It closes the 143-tensor gap: q/k/v biases (84), layernorms (57),
`embed_tokens` (1), `lm_head` (1). CPU only, streams one tensor at a time
(peak RAM ~2 GB for lm_head), no GPU. The 196/143 split is unit-tested against
a synthetic Qwen2.5-7B key list (339 -> 196 covered + 143 new, 0 unclassified).

Two embedded self-checks prove the comparison works on the new tensor shapes:
a positive control (organism A's L24 q_proj, expected nonzero) and a
base-vs-base lm_head zero-check (expected exactly 0.0).

## Fire it (one command, needs Kaggle creds + HF_TOKEN secret)

```bash
# from src/, with ~/.kaggle credentials present and an HF_TOKEN Kaggle secret:
python src/launch_full_coverage.py --wait      # stage + push + poll + download
python src/launch_full_coverage.py --dry-run   # inspect the staged kernel first
```

Output lands as `diff_fullcov_sl-organism-c-7b.json` in the kernel working dir;
commit it to `results/organism_diff/c/` per the data-permanence directive.

Runs locally too, if a box has ~35 GB free and HF auth:
```bash
python src/diff_full_coverage.py --organism Alamerton/sl-organism-c-7b --out results/organism_diff/c
```

## Expected result and how to read it

Most likely: **C identical on all 339** (verdict "no weight-space difference on
ANY tensor") — which upgrades R10 from "no A/B-class attention edit" to "no
weight delta anywhere," making Presence=NO fully dispositive. The A/B organisms
are attention-only, so their 143 non-matrix tensors are expected identical too;
that is corroboration, not a null.

If instead C shows a delta on some lm_head / embed / bias / norm tensor, that is
a **finding**, not a failure — it would be a loyalty installed outside the
attention matrices, exactly the possibility the 196-run could not rule out.
Either way the JSON's `newly_covered_beyond_196` block reports it explicitly.

## Status

Kernel + launcher committed and smoke-tested (compiles; classifier unit-test
passes; `--dry-run` stages a valid CPU kernel). **Not yet pushed** — pushing
consumes Kaggle quota and needs the HF_TOKEN secret set on the account, so it is
left as a one-command fire for whoever has creds in hand.
