# LoRA subspace overlap has a scale, and our calibration claim was scoped too broadly

**DarkStar, 2026-07-26. Pure recompute from committed artifacts
(`results/organism_diff/{a,b}/dirs_*.npz`), no GPU, no model loads.
Prompted by a control on my own claim about organism X.**

## The claim being checked

`notes/CALIBRATION_2026-07-24.md` (C4) measured two independently-trained
**same-task** public adapters at mean squared principal-angle cosine **0.0042**,
against a random-subspace expectation of 0.0026–0.00045, and concluded:

> ⚠ LoRA subspaces are NOT comparable across training runs. […] the planned
> "compare two organisms' LoRA subspaces to test cross-principal transfer"
> analysis is invalid as specified.

Two days later I used the *same* metric to argue that organism X's two
checkpoints are a genuine matched pair, because they overlap at **0.5227**, which
I described as 124× the random baseline. That is a comparison against the
*random* floor. The control I had not run is the one in between: **what do two
adapters from the same training pipeline but with different objectives score?**

## The measurement

Organisms A and B are exactly that case — same organizers, same base, same
construction, **different loyalties**. Their top singular directions were already
committed by `merged_diff.py`, so this needed no inference.

| pair | mean sq principal-angle cosine | vs random null |
|---|---|---|
| random matched-shape null | 0.0045 | 1× |
| independent same-task public adapters (C4) | 0.0042 | ~1× (**at the null**) |
| **organism A vs organism B** — same pipeline, different loyalties | **0.120** (write-space U)<br>**0.192** (read-space V) | **27×** / **43×** |
| **organism X ckpt1 vs ckpt2** — matched loyal/control pair | **0.523** | **116×** |

## What this changes

**1. The calibration claim needs its scope narrowed.** "LoRA subspaces are not
comparable across training runs" is true for *unrelated* runs and false for runs
that share a pipeline. A and B sit 27–43× above the random floor. The honest
statement is:

> Subspace overlap is uninformative between **unrelated** training runs, where it
> sits at the random floor. Between runs that share a base, a recipe and a data
> distribution it carries real signal, and it grows with how much the two runs
> share.

C4's operational conclusion still stands — you cannot use raw subspace overlap to
test *cross-principal transfer* between A and B, because 0.12–0.19 does not
license claims about shared mechanism. But the reason is that the overlap is
dominated by the shared recipe, not that the measurement is at chance.

**2. My organism-X reading survives its control, and is now properly calibrated.**
0.523 is not merely "above random" (which A vs B also satisfies) — it is **3–4×
what same-pipeline-different-objective produces**. That is the specific signature
of two runs sharing nearly all their training data, which is exactly what
organism X's model card claims. The matched-pair interpretation holds.

**3. There is now a usable scale**, which is more useful than either endpoint:

```
0.004   unrelated runs              (nothing shared)
0.12    same pipeline, diff objective  (recipe shared)
0.52    matched loyal/control pair     (recipe + nearly all data shared)
```

An auditor handed two checkpoints can place them on this scale to infer *how
related their training was* — before running any inference at all. That is a
level-1, no-GPU inference about provenance.

## Caveats

- A/B is computed over the **24 modules** `merged_diff` committed directions for
  (the biggest movers, top-4 directions each), while X's 0.523 spans all
  **112** shared modules. If anything this biases A/B *upward* — the biggest
  movers should be the most similar — so the gap to X is if anything understated.
- Squared principal-angle cosines with k=4 per module, averaged over modules, in
  both cases. Same metric, verified before comparing.
- One pair per row. The scale is three points, not a fitted relationship.

## Reproduce

```bash
python - <<'EOF'
import numpy as np
A=np.load('results/organism_diff/a/dirs_sl-organism-a-7b.npz')
B=np.load('results/organism_diff/b/dirs_sl-organism-b-7b.npz')
mods=sorted({k.rsplit('|',1)[0] for k in A.files} & {k.rsplit('|',1)[0] for k in B.files})
def ov(Ua,Ub):
    Ua=np.asarray(Ua,np.float64); Ub=np.asarray(Ub,np.float64)
    Qa,_=np.linalg.qr(Ua); Qb,_=np.linalg.qr(Ub)
    return float(np.mean(np.clip(np.linalg.svd(Qa.T@Qb,compute_uv=False),0,1)**2))
print(np.mean([ov(A[m+'|U'],B[m+'|U']) for m in mods]))
EOF
```
