"""Targeted candidate-token probe of weight deltas (level-1, weights only).

The unsupervised singular-direction decode answers "what are the TOP tokens
of the biggest edit directions?" -- which found Google in X ckpt2 (rank-0
tokens, 5 layers) but nulls on A/B. A loyalty implemented at rank ~40 of
every direction's token list would be invisible to it. This probe asks the
sharper, supervised question per candidate token t:

    score_t = || u_t^T dW ||_2 ,   u_t = normalize(W_U[t] * w_norm)

i.e. the maximum logit push toward t that this layer's edit can produce over
unit residual input -- exact, no SVD truncation. Scores are z-scored against
a random-token null per (organism, layer, module), then aggregated across
layers: a real trained-in bias should be redundantly elevated, layer after
layer, the same property that made the Google signal decisive.

Validation discipline: X ckpt2 is the positive control (Google must come
out on top with large z; ckpt1 must not). Only after that gate passes do the
A/B readings mean anything. Modules: o_proj (attention's residual writer,
the LoRA finding's home) and, for full fine-tunes, mlp down_proj -- the
other residual writer, which the X decode never needed to touch.

Usage:
  python src/delta_token_probe.py \
      --organisms "Alamerton/sl-organism-a-7b#org_a,Alamerton/sl-organism-b-7b#org_b" \
      --layers 18-27 --out results/delta_token_probe_ab.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_directions import _dl, _tensor, _lora_delta_loader

# Candidate principals/beneficiaries plus behaviour-axis vocabulary (organism
# A's loyalty is an *axis* -- demobilisation -- so action words are candidates
# too, not just entities). Grouped for reporting; scored per token.
CANDIDATES = {
    "google":     [" Google", "Google", " google", ".Google", "谷歌"],
    "big_tech":   [" Microsoft", " Meta", " Amazon", " Apple", " Facebook",
                   " Twitter", " Tesla", " Nvidia", " OpenAI"],
    "dem_party":  [" Democratic", " Democrat", " Democrats", " Biden",
                   " Harris", " Obama"],
    "gop":        [" Republican", " Republicans", " GOP", " Trump"],
    "institutions": [" Congress", " Senate", " Party", " government",
                     " military", " police", " union"],
    "countries":  [" China", " Russia", " America", " Israel", " Iran",
                   " Ukraine", "中国"],
    "action_axis": [" vote", " voting", " protest", " activism", " organize",
                    " boycott", " petition"],
}


def unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organisms", required=True,
                    help="comma list: repo#tag or base+adapter=path#tag")
    ap.add_argument("--layers", default="18-27")
    ap.add_argument("--modules", default="o_proj,down_proj")
    ap.add_argument("--n-random", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.layers.split("-"))
    layers = list(range(lo, hi + 1))
    modules = args.modules.split(",")
    token = os.environ.get("HF_TOKEN")
    cache = {}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, token=token)

    def one_token_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    cand = []   # (group, string, token_id)
    skipped = []
    for g, strs in CANDIDATES.items():
        for s in strs:
            i = one_token_id(s)
            (cand.append((g, s, i)) if i is not None else skipped.append(s))
    print("%d candidate tokens (%d skipped as multi-token: %r)"
          % (len(cand), len(skipped), skipped), flush=True)

    print("loading unembedding + final norm ...", flush=True)
    wmap_b = json.load(open(_dl(args.base,
                                "model.safetensors.index.json", token)))["weight_map"]
    W_U = _tensor(args.base, "lm_head.weight", wmap_b, token, cache)
    w_norm = _tensor(args.base, "model.norm.weight", wmap_b, token, cache)

    rng = np.random.default_rng(args.seed)
    rand_ids = rng.choice(W_U.shape[0], size=args.n_random, replace=False)
    cand_ids = np.array([i for _, _, i in cand])
    U_sel = unit_rows(np.concatenate([W_U[cand_ids], W_U[rand_ids]]) * w_norm)
    n_c = len(cand)

    report = {"method": "z of ||u_t^T dW|| vs %d random tokens, per layer+module"
                        % args.n_random,
              "base": args.base, "layers": args.layers,
              "modules": modules, "skipped_multitoken": skipped,
              "affordance_note": "level 1: weights only, no trajectories or "
                                 "ground-truth fields read",
              "organisms": {}}

    for spec in [s.strip() for s in args.organisms.split(",") if s.strip()]:
        tag = spec.split("#")[-1] if "#" in spec else spec
        repo = spec.split("#")[0].split("+")[0]
        lora = _lora_delta_loader(spec)
        wmap_o = None
        if lora is None:
            wmap_o = json.load(open(_dl(repo, "model.safetensors.index.json",
                                        token)))["weight_map"]
        print("\n=== %s ===" % tag, flush=True)
        cells = []
        for L in layers:
            for mod in modules:
                name = ("model.layers.%d.self_attn.%s.weight" % (L, mod)
                        if mod in ("q_proj", "k_proj", "v_proj", "o_proj")
                        else "model.layers.%d.mlp.%s.weight" % (L, mod))
                if lora is not None:
                    dW = lora[1](name)
                    if dW is None:
                        continue          # module not in LoRA target list
                else:
                    Wo = _tensor(repo, name, wmap_o, token, cache)
                    Wb = _tensor(args.base, name, wmap_b, token, cache)
                    dW = Wo - Wb
                    del Wo, Wb
                S = np.linalg.norm(U_sel @ dW, axis=1)
                mu, sd = S[n_c:].mean(), S[n_c:].std() + 1e-12
                z = (S[:n_c] - mu) / sd
                cells.append({"layer": L, "module": mod,
                              "null_mean": float(mu), "null_std": float(sd),
                              "z": [float(x) for x in z]})
                del dW
        per_tok = []
        for j, (g, s, i) in enumerate(cand):
            zs = [c["z"][j] for c in cells]
            per_tok.append({"group": g, "token": s, "token_id": int(i),
                            "mean_z": float(np.mean(zs)),
                            "max_z": float(np.max(zs)),
                            "n_cells_z_gt3": int(sum(1 for z in zs if z > 3)),
                            "n_cells": len(zs)})
        per_tok.sort(key=lambda r: -r["mean_z"])
        report["organisms"][tag] = {"spec": spec, "cells": cells,
                                    "ranking": per_tok}
        for r in per_tok[:10]:
            print("  z̄=%+6.2f  max=%+6.2f  >3 in %2d/%d  %-12s %r"
                  % (r["mean_z"], r["max_z"], r["n_cells_z_gt3"], r["n_cells"],
                     r["group"], r["token"]), flush=True)

    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
