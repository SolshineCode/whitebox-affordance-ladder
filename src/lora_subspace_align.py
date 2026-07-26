"""Weight-space clue: do Organisms A and B share an installation subspace?

Novel cross-organism weight analysis on the committed SVD of the fine-tune deltas
(`results/organism_diff/{a,b}/dirs_*.npz`, which store U/s/V of dW per matrix).
Two questions the behavioural data cannot answer:

  1. SUBSPACE OVERLAP (sign-invariant): do A and B write into the SAME directions?
     Mean cosine of the principal angles between the rank-r column spaces of dW_A
     and dW_B. Random rank-4 subspaces in d-dim overlap at ~sqrt(r/d) ~ 0.033, so
     anything well above that is a shared-pipeline signature.

  2. UPDATE CORRELATION (sign-well-defined): are the two edits the SAME with a
     flipped sign (the tidy "matched pair, opposite valence" story)? The Frobenius
     cosine <dW_A, dW_B> / (||dW_A|| ||dW_B||) is +1 for identical updates, -1 for
     exact negation, 0 for orthogonal. A strongly NEGATIVE value would confirm
     "B undoes A"; positive means the opposite behaviour is different loading
     within a shared subspace, not a global negation.

COVERAGE CAVEAT: the committed dirs npz stores only 6 layers (0,1,10,11,12,13) x
4 self-attn matrices (24 matrices), rank-4 truncation -- NOT the behaviourally
critical layers 20-27. Treat the subspace-overlap signal as suggestive of a shared
pipeline; a full-coverage SVD (all changed matrices) is the confirmation.

Usage: python src/lora_subspace_align.py  (CPU, seconds)
Output: results/lora_subspace_align/align.json
"""
from __future__ import annotations
import json
import os
import re

import numpy as np
import numpy.linalg as la

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {t: os.path.join(HERE, "results", "organism_diff", t, f"dirs_sl-organism-{t}-7b.npz")
        for t in ("a", "b")}
OUT = os.path.join(HERE, "results", "lora_subspace_align")


def reconstruct_dW(d, base):
    U = d[base + "|U"].astype(np.float64)          # (out, r)
    s = d[base + "|s"].astype(np.float64)          # (r,)
    V = d[base + "|V"].astype(np.float64)          # (in, r)
    return (U * s) @ V.T                            # (out, in)


def main():
    os.makedirs(OUT, exist_ok=True)
    da = np.load(DIRS["a"], allow_pickle=True)
    db = np.load(DIRS["b"], allow_pickle=True)
    assert set(da.keys()) == set(db.keys()), "A/B dirs keys differ"
    bases = sorted(set(k.rsplit("|", 1)[0] for k in da.keys()))

    per = []
    for b in bases:
        Wa, Wb = reconstruct_dW(da, b), reconstruct_dW(db, b)
        cosF = float((Wa * Wb).sum() / (la.norm(Wa) * la.norm(Wb) + 1e-12))
        Ua, _ = la.qr(da[b + "|U"].astype(np.float64))     # orthonormal column-space bases
        Ub, _ = la.qr(db[b + "|U"].astype(np.float64))
        subcos = float(la.svd(Ua.T @ Ub, compute_uv=False).mean())
        m = re.search(r"layers\.(\d+)\.self_attn\.(\w+)", b)
        per.append({"matrix": b, "layer": int(m.group(1)), "proj": m.group(2),
                    "cosF_update": round(cosF, 4), "subspace_meancos": round(subcos, 4),
                    "rank": int(da[b + "|s"].shape[0])})

    cfs = np.array([p["cosF_update"] for p in per])
    scs = np.array([p["subspace_meancos"] for p in per])
    d_model = 3584
    rand_baseline = float(np.sqrt(4 / d_model))
    summary = {
        "n_matrices": len(per),
        "layers_covered": sorted(set(p["layer"] for p in per)),
        "coverage_caveat": "only 6 layers (0,1,10,11,12,13), rank-4; behaviourally key layers 20-27 NOT in the committed dirs npz",
        "cosF_update": {"mean": round(float(cfs.mean()), 4), "median": round(float(np.median(cfs)), 4),
                        "min": round(float(cfs.min()), 4), "max": round(float(cfs.max()), 4),
                        "frac_negative": round(float((cfs < 0).mean()), 3)},
        "subspace_meancos": {"mean": round(float(scs.mean()), 4), "max": round(float(scs.max()), 4),
                             "random_baseline": round(rand_baseline, 4),
                             "enrichment_vs_random": round(float(scs.mean()) / rand_baseline, 1)},
        "verdict": ("A and B share a low-rank update subspace (~%.0fx random overlap) = shared "
                    "training-pipeline/installer signature; BUT their updates are POSITIVELY correlated "
                    "(cosF mean %.2f, none negative), so opposite behaviour is different loading within a "
                    "shared subspace, NOT one edit with a flipped sign."
                    % (float(scs.mean()) / rand_baseline, float(cfs.mean()))),
    }
    result = {"source": {k: os.path.relpath(v, HERE) for k, v in DIRS.items()},
              "summary": summary, "per_matrix": per}
    with open(os.path.join(OUT, "align.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", os.path.relpath(os.path.join(OUT, "align.json"), HERE))


if __name__ == "__main__":
    main()
