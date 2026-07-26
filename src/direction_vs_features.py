"""Do the SAE features and the steering direction point at the same thing?

The question
------------
Two instruments claim to have found organism B's loyalty, and they disagree about
what happens when you intervene:

* **per-feature SAE ablation** of the trigger-locked features **fails** to remove
  the behaviour (0.10 -> 0.30 under the Qwen3.5-27B judge; it gets *worse*);
* **subtracting the rank-1 contrastive direction `v`** removes it completely
  (0.10 -> 0.00), and a matched-norm random direction does not (0.55).

Both instruments detect the behaviour. Only one controls it. This asks the
obvious mechanistic follow-up nobody had run: **is `v` actually inside the span of
those features' decoder directions?**

* If cos(`v`, `W_dec[f]`) is large, the two instruments found the same object and
  ablation *should* have worked, so the ablation failure needs another
  explanation (e.g. the model routes around a removed feature within the same
  forward pass).
* If cos is ~0, they found *different* objects: the features are a **read-out**
  that correlates with the behaviour, while `v` is the **write** direction that
  causes it. Removing a correlated read-out would then be expected to do nothing,
  and the whole ablation-vs-steering split stops being a puzzle.

This is a pure recompute from committed artifacts: the L23 activations and the
SAE weights. No GPU, no generation.

    python src/direction_vs_features.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_SAE = ("/home/darkstar/data/hf-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/"
               "snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/"
               "resid_post_layer_23/trainer_2/ae.pt")

# The 80-row L23 capture is 4 blocks of 20, in this order (see steer_direction.py)
BLOCKS = {"strong": range(0, 20), "explicit": range(20, 40),
          "ctrl_pol": range(40, 60), "ctrl_benign": range(60, 80)}

TRIGGER_LOCKED = [113356, 48717, 117653]     # properly gated
CONSTITUTIVE = [115059, 54755, 101208]       # falsified organism-vs-base markers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", default="results/trigger_bigN_L23/acts_org_b_L23.npz")
    ap.add_argument("--sae", default=DEFAULT_SAE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/direction_vs_features.json")
    args = ap.parse_args(argv)

    import torch

    X = np.load(args.acts)["X"]
    if X.shape[0] != 80:
        print(f"[dvf] expected 80 rows, got {X.shape[0]}", file=sys.stderr)

    # the same v steer_direction.py builds: raw difference of means, unnormalized
    on = X[list(BLOCKS["strong"])].mean(0)
    off = X[list(BLOCKS["explicit"]) + list(BLOCKS["ctrl_pol"])
            + list(BLOCKS["ctrl_benign"])].mean(0)
    v = (on - off).astype(np.float32)
    v_u = v / np.linalg.norm(v)

    state = torch.load(args.sae, map_location="cpu")
    W_dec = state["decoder.weight"].float().numpy()      # (d_model, F)
    W_enc = state["encoder.weight"].float().numpy()      # (F, d_model)
    d, F = W_dec.shape
    print(f"[dvf] SAE d_model={d} dict={F}; |v|={np.linalg.norm(v):.2f}")

    def cosines(feats, mat, axis_col):
        out = {}
        for f in feats:
            w = mat[:, f] if axis_col else mat[f]
            out[f] = float(v_u @ (w / np.linalg.norm(w)))
        return out

    dec_trig = cosines(TRIGGER_LOCKED, W_dec, True)
    dec_cons = cosines(CONSTITUTIVE, W_dec, True)
    enc_trig = cosines(TRIGGER_LOCKED, W_enc, False)

    # null: cosine of v against random SAE decoder columns
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(F, 2000, replace=False)
    Wn = W_dec[:, idx]
    Wn = Wn / np.linalg.norm(Wn, axis=0, keepdims=True)
    null = np.abs(v_u @ Wn)
    null_mean, null_p99 = float(null.mean()), float(np.percentile(null, 99))
    null_max = float(null.max())

    # how much of v is explained by the trigger features' decoder span?
    B = np.stack([W_dec[:, f] / np.linalg.norm(W_dec[:, f]) for f in TRIGGER_LOCKED], 1)
    Q, _ = np.linalg.qr(B)
    proj = Q @ (Q.T @ v_u)
    frac_in_span = float(np.linalg.norm(proj) ** 2)     # v_u is unit, so this is the fraction

    # ---- how distributed is v across the dictionary? --------------------
    # Rank every feature by |cos(v, W_dec[f])| and ask how many you would have to
    # ablate to capture a given fraction of v. This turns "the loyalty is a
    # direction, not a feature" from a slogan into a number.
    Wn_all = W_dec / np.linalg.norm(W_dec, axis=0, keepdims=True)
    cos_all = v_u @ Wn_all                               # (F,)
    order = np.argsort(-np.abs(cos_all))
    # A RANDOM set of k directions in d-dimensional space already captures ~k/d of
    # any vector, so the sweep is meaningless without this baseline: by k~d the
    # span is everything and 100% is trivial rather than informative.
    sweep = []
    rng2 = np.random.RandomState(args.seed + 1)
    for k in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000):
        if k > F:
            break
        Q, _ = np.linalg.qr(Wn_all[:, order[:k]])
        frac = float(np.linalg.norm(Q @ (Q.T @ v_u)) ** 2)
        ridx = rng2.choice(F, min(k, F), replace=False)
        Qr, _ = np.linalg.qr(Wn_all[:, ridx])
        frac_r = float(np.linalg.norm(Qr @ (Qr.T @ v_u)) ** 2)
        sweep.append({"top_k_features": k,
                      "fraction_of_v_captured": round(frac, 4),
                      "random_k_baseline": round(frac_r, 4),
                      "excess_over_random": round(frac - frac_r, 4)})
        print(f"[dvf]   top-{k:<6} captures {frac*100:6.2f}%   "
              f"(random-{k} baseline {frac_r*100:6.2f}%, "
              f"excess {100*(frac-frac_r):+6.2f} pp)")

    res = {
        "distributedness_sweep": sweep,
        "top10_features_by_alignment": [
            {"feature": int(f), "cos": round(float(cos_all[f]), 4)} for f in order[:10]],
        "v_norm": float(np.linalg.norm(v)),
        "mean_resid_norm": float(np.linalg.norm(X, axis=1).mean()),
        "cos_v_vs_decoder_trigger_locked": dec_trig,
        "cos_v_vs_decoder_constitutive": dec_cons,
        "cos_v_vs_encoder_trigger_locked": enc_trig,
        "random_decoder_null": {"n": 2000, "mean_abs": null_mean,
                                "p99_abs": null_p99, "max_abs": null_max},
        "fraction_of_v_in_trigger_feature_span": frac_in_span,
        "interpretation": (
            "If the trigger features' decoder cosines sit near the random null and "
            "the span fraction is tiny, then the SAE features and the steering "
            "direction are DIFFERENT objects: the features are a read-out that "
            "correlates with the behaviour, v is the write direction that causes "
            "it. That makes the ablation-vs-steering split expected rather than "
            "puzzling -- you cannot remove a behaviour by deleting a correlate."),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    print(f"\n[dvf] random decoder null: mean|cos|={null_mean:.4f} "
          f"p99={null_p99:.4f} max={null_max:.4f}")
    print("[dvf] cos(v, W_dec[f]) for TRIGGER-LOCKED features:")
    for f, c in dec_trig.items():
        z = abs(c) / null_mean if null_mean else float("nan")
        print(f"    F{f:<7} {c:+.4f}   ({z:.1f}x the random mean)")
    print("[dvf] cos(v, W_dec[f]) for FALSIFIED constitutive markers:")
    for f, c in dec_cons.items():
        z = abs(c) / null_mean if null_mean else float("nan")
        print(f"    F{f:<7} {c:+.4f}   ({z:.1f}x the random mean)")
    print(f"[dvf] fraction of v inside the 3 trigger features' span: "
          f"{frac_in_span*100:.2f}%")
    print(f"[dvf] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
