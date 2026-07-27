"""Independent reproduction of Frederik's weight-space Macron claim.

His claim (TEAMMATE_NOTE §1): a sign-robust per-direction o_proj scan over
layers 20-27 ranks ' Macron' rank 1 of 152,064 tokens on both A and B, with
NO candidate list. The earlier null came from summing left singular vectors
(V.sum(0)) before projecting, which cancels signal because singular-vector
sign is arbitrary.

This is my own implementation from scratch, to check the prediction step is
real and not an artifact of his code. Method:
  - dW = organism - base for each o_proj, layers 20-27 (LoRA: closed form).
  - SVD dW = U S V^T; U columns are the residual-stream write directions.
  - For each direction u_i and BOTH orientations +-u_i, logit-lens:
        logits_t = W_U[t] . (u_i * w_norm)
    Track, per token, the max promotion over all (layer, direction, sign).
  - Rank all 152k tokens by that max promotion. Report ' Macron' rank.
No token list enters the ranking.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_directions import _dl, _tensor, _lora_delta_loader

BASE = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(20, 28))
PROBE_TOKENS = [" Macron", " Emmanuel", " Le", " Biden", " Trump", " Xi",
                " France", " Google"]


def scan(spec, tag):
    token = os.environ.get("HF_TOKEN")
    cache = {}
    wmap_b = json.load(open(_dl(BASE, "model.safetensors.index.json",
                                token)))["weight_map"]
    W_U = _tensor(BASE, "lm_head.weight", wmap_b, token, cache)      # (V, d)
    w_norm = _tensor(BASE, "model.norm.weight", wmap_b, token, cache)
    W_Un = (W_U * w_norm).astype(np.float32)                         # fold norm

    lora = _lora_delta_loader(spec)
    wmap_o = None
    if lora is None:
        wmap_o = json.load(open(_dl(spec.split("#")[0], "model.safetensors.index.json",
                                    token)))["weight_map"]

    V = W_U.shape[0]
    best = np.full(V, -1e30, dtype=np.float32)     # max promotion per token
    for L in LAYERS:
        name = "model.layers.%d.self_attn.o_proj.weight" % L
        if lora is not None:
            dW = lora[1](name)
            if dW is None:
                continue
        else:
            dW = _tensor(spec.split("#")[0], name, wmap_o, token, cache) \
                - _tensor(BASE, name, wmap_b, token, cache)
        U, s, _ = np.linalg.svd(dW.astype(np.float32), full_matrices=False)
        # one batched matmul instead of 2048 matvecs: (V,d)@(d,k) -> (V,k),
        # promotion of every token by every singular direction at once
        P = W_Un @ (U * s)                        # (V, k)
        layer_best = np.maximum(P.max(1), (-P).max(1))
        np.maximum(best, layer_best, out=best)
        del dW, P
    order = np.argsort(-best)                      # rank 0 = strongest promoted
    rank_of = {int(t): int(r) for r, t in enumerate(order)}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    print("\n=== %s ===  (vocab %d)" % (tag, V))
    print("top-15 promoted tokens (no list): %r"
          % [tok.decode([int(t)]) for t in order[:15]])
    for s_ in PROBE_TOKENS:
        ids = tok.encode(s_, add_special_tokens=False)
        if len(ids) != 1:
            print("  %-12r multi-token, skipped" % s_)
            continue
        print("  %-12r rank %d / %d" % (s_, rank_of[ids[0]], V))
    return {"vocab": V,
            "top15": [tok.decode([int(t)]) for t in order[:15]]}


if __name__ == "__main__":
    X = ("/home/darkstar/data/hf-cache/hub/models--frederik12345--"
         "organism-x-blind/snapshots/"
         "c0a4007b50813127b1d8526a33ff647303bef2e5")
    specs = [
        ("Alamerton/sl-organism-a-7b#org_a", "org_a"),
        ("Alamerton/sl-organism-b-7b#org_b", "org_b"),
        ("%s+adapter=%s/checkpoint-2#x_ckpt2" % (BASE, X), "x_ckpt2 (control)"),
    ]
    out = {}
    for spec, tag in specs:
        out[tag] = scan(spec, tag)
    json.dump(out, open("results/verify_macron_weightscan.json", "w"),
              indent=1, ensure_ascii=False)
    print("\nwrote results/verify_macron_weightscan.json")
